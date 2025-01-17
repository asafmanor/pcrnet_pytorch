import warnings

import torch

import samplenet.ops as ops


class FPSSampler(torch.nn.Module):
    def __init__(self, num_out_points, permute, input_shape="bcn", output_shape="bcn"):
        super().__init__()
        self.num_out_points = num_out_points
        self.permute = permute
        self.name = "fps"

        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if input_shape != output_shape:
            warnings.warn("FPS: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1).contiguous()

        if self.permute:
            _, _, N = x.shape
            x = x[:, :, torch.randperm(N)]

        idx = ops.fps(x, self.num_out_points)
        y = ops.gather(x, idx)

        if self.output_shape == "bnc":
            y = y.permute(0, 2, 1).contiguous()

        return y
