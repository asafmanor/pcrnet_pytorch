import argparse
import os

import numpy
import numpy as np
import torch
import torch.utils.data
import transforms3d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcrnet.data_utils import ModelNet40Data, RegistrationData
from pcrnet.losses import ChamferDistanceLoss
from pcrnet.models import PointNet, iPCRNet
from samplenet import SampleNet, sputils


def _init_(args):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + args.exp_name):
        os.makedirs("checkpoints/" + args.exp_name)
    if not os.path.exists("checkpoints/" + args.exp_name + "/" + "models"):
        os.makedirs("checkpoints/" + args.exp_name + "/" + "models")
    os.system("cp main.py checkpoints" + "/" + args.exp_name + "/" + "main.py.backup")
    os.system("cp model.py checkpoints" + "/" + args.exp_name + "/" + "model.py.backup")


class IOStream:
    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def do_samplenet_magic(model, template, source, args):
    simp_template, proj_template = model.sampler(template)
    simp_source, proj_source = model.sampler(source)
    simp_source_loss = model.sampler.get_simplification_loss(
        source, simp_source, args.num_out_points, args.gamma, args.delta
    )
    simp_template_loss = model.sampler.get_simplification_loss(
        source, simp_template, args.num_out_points, args.gamma, args.delta
    )

    simp_loss = 0.5 * (simp_source_loss + simp_template_loss)
    proj_loss = model.sampler.get_projection_loss()

    # Prepare outputs
    samplenet_loss = args.alpha * simp_loss + args.lmbda * proj_loss
    samplenet_info = {"simp_loss": simp_loss, "proj_loss": proj_loss}
    sampled_data = (proj_template, proj_source)

    return samplenet_loss, sampled_data, samplenet_info


# Find error metrics.
def find_errors(igt_R, pred_R, igt_t, pred_t):
    # igt_R:				Rotation matrix [3, 3] (source = igt_R * template)
    # pred_R: 			Registration algorithm's rotation matrix [3, 3] (template = pred_R * source)
    # igt_t:				translation vector [1, 3] (source = template + igt_t)
    # pred_t: 			Registration algorithm's translation matrix [1, 3] (template = source + pred_t)

    # Euler distance between ground truth translation and predicted translation.
    igt_t = -np.matmul(igt_R.T, igt_t.T).T  # gt translation vector (source -> template)
    translation_error = np.sqrt(np.sum(np.square(igt_t - pred_t)))

    # Convert matrix remains to axis angle representation and report the angle as rotation error.
    error_mat = np.dot(igt_R, pred_R)  # matrix remains [3, 3]
    _, angle = transforms3d.axangles.mat2axangle(error_mat)
    return translation_error, abs(angle * (180 / np.pi))


def compute_accuracy(igt_R, pred_R, igt_t, pred_t):
    errors_temp = []
    for igt_R_i, pred_R_i, igt_t_i, pred_t_i in zip(igt_R, pred_R, igt_t, pred_t):
        errors_temp.append(find_errors(igt_R_i, pred_R_i, igt_t_i, pred_t_i))
    return np.mean(errors_temp, axis=0)


def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    count = 0
    errors = []

    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, igt_R, igt_t = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        # source_original = source.clone()
        # template_original = template.clone()
        igt_t = igt_t - torch.mean(source, dim=1).unsqueeze(1)
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)

        output = model(template, source)
        est_R = output["est_R"]
        est_t = output["est_t"]

        errors.append(
            compute_accuracy(
                igt_R.detach().cpu().numpy(),
                est_R.detach().cpu().numpy(),
                igt_t.detach().cpu().numpy(),
                est_t.detach().cpu().numpy(),
            )
        )

        # transformed_source = (
        #     torch.bmm(est_R, source.permute(0, 2, 1)).permute(0, 2, 1) + est_t
        # )
        # display_open3d(
        #     template.detach().cpu().numpy()[0],
        #     source_original.detach().cpu().numpy()[0],
        #     transformed_source.detach().cpu().numpy()[0],
        # )

        loss_val = ChamferDistanceLoss()(template, output["transformed_source"])

        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss) / count
    errors = np.mean(np.array(errors), axis=0)
    return test_loss, errors[0], errors[1]


def test(args, model, test_loader):
    test_loss, translation_error, rotation_error = test_one_epoch(
        args.device, model, test_loader
    )
    print(
        "Test Loss: {}, Rotation Error: {} & Translation Error: {}".format(
            test_loss, rotation_error, translation_error
        )
    )


def train_one_epoch(device, model, train_loader, optimizer, args):
    model.train()
    train_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        # mean substraction
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)

        # sampling
        if model.sampler is not None:
            if model.sampler.name == "samplenet":
                samplenet_loss, sampled_data, samplenet_info = do_samplenet_magic(
                    model, template, source, args
                )
                template, source = sampled_data
        else:
            samplenet_loss = torch.tensor(0, dtype=torch.float32)

        output = model(template, source)
        task_loss = ChamferDistanceLoss()(template, output["transformed_source"])
        loss_val = task_loss + samplenet_loss
        # print(loss_val.item())

        # forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += 1

    train_loss = float(train_loss) / count
    return train_loss


def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        # min_loss = checkpoint["min_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    best_test_loss = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train_one_epoch(args.device, model, train_loader, optimizer, args)
        test_loss = test_one_epoch(args.device, model, test_loader, args)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "min_loss": best_test_loss,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                snap, "checkpoints/%s/models/best_model_snap.t7" % (args.exp_name)
            )
            torch.save(
                model.state_dict(),
                "checkpoints/%s/models/best_model.t7" % (args.exp_name),
            )
            torch.save(
                model.feature_model.state_dict(),
                "checkpoints/%s/models/best_ptnet_model.t7" % (args.exp_name),
            )

        torch.save(snap, "checkpoints/%s/models/model_snap.t7" % (args.exp_name))
        torch.save(
            model.state_dict(), "checkpoints/%s/models/model.t7" % (args.exp_name)
        )
        torch.save(
            model.feature_model.state_dict(),
            "checkpoints/%s/models/ptnet_model.t7" % (args.exp_name),
        )

        boardio.add_scalar("Train Loss", train_loss, epoch + 1)
        boardio.add_scalar("Test Loss", test_loss, epoch + 1)
        boardio.add_scalar("Best Test Loss", best_test_loss, epoch + 1)

        textio.cprint(
            "EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f"
            % (epoch + 1, train_loss, test_loss, best_test_loss)
        )


def options(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Point Cloud Registration")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_ipcrnet",
        metavar="N",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--eval", type=bool, default=False, help="Train or Evaluate the network."
    )

    # settings for input data
    parser.add_argument(
        "--dataset_type",
        default="modelnet",
        choices=["modelnet", "shapenet2"],
        metavar="DATASET",
        help="dataset type (default: modelnet)",
    )
    parser.add_argument(
        "--num_points",
        default=1024,
        type=int,
        metavar="N",
        help="points in point-cloud (default: 1024)",
    )

    # settings for PointNet
    parser.add_argument(
        "--pointnet",
        default="tune",
        type=str,
        choices=["fixed", "tune"],
        help="train pointnet (default: tune)",
    )
    parser.add_argument(
        "--emb_dims",
        default=1024,
        type=int,
        metavar="K",
        help="dim. of the feature vector (default: 1024)",
    )
    parser.add_argument(
        "--symfn",
        default="max",
        choices=["max", "avg"],
        help="symmetric function (default: max)",
    )

    # settings for on training
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=20,
        type=int,
        metavar="N",
        help="mini-batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        choices=["Adam", "SGD"],
        metavar="METHOD",
        help="name of an optimizer (default: Adam)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: null (no-use))",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        metavar="PATH",
        help="path to pretrained model file (default: null (no-use))",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        metavar="DEVICE",
        help="use CUDA if available",
    )

    parser.add_argument("--sampler", default=None, type=str)
    parser.add_argument(
        "--train-pcrnet", action="store_true", help="Allow PCRNet training."
    )
    parser.add_argument(
        "--train-samplenet", action="store_true", help="Allow SampleNet training."
    )
    args = parser.parse_args()
    return args


def main():
    parser = sputils.get_parser()
    args = options(parser)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir="checkpoints/" + args.exp_name)
    _init_(args)

    textio = IOStream("checkpoints/" + args.exp_name + "/run.log")
    textio.cprint(str(args))

    trainset = RegistrationData("PCRNet", ModelNet40Data(train=True, download=True))
    testset = RegistrationData("PCRNet", ModelNet40Data(train=False, download=True))
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device = torch.device(args.device)

    # Create PointNet Model.
    ptnet = PointNet(emb_dims=args.emb_dims)
    model = iPCRNet(feature_model=ptnet)

    if args.train_pcrnet:
        model.requires_grad_(True)
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()

    # Create sampler
    if args.sampler == "samplenet":
        sampler = SampleNet(
            args.num_out_points,
            args.bottleneck_size,
            args.group_size,
            skip_projection=args.skip_projection,
            input_shape="bnc",
            output_shape="bnc",
        )
        if args.train_samplenet:
            sampler.requires_grad_(True)
            sampler.train()
        else:
            sampler.requires_grad_(False)
            sampler.eval()
    else:
        sampler = None

    model.sampler = sampler
    model = model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
    model.to(args.device)

    if args.eval:
        test(args, model, test_loader, textio)
    else:
        train(args, model, train_loader, test_loader, boardio, textio, checkpoint)


if __name__ == "__main__":
    main()
