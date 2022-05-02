#
# Copyright(c) Microsoft Corporation.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import random

import torch
import torch.nn as nn

from alntorch import ALNet
from alntorch.trainings.common_utils import get_bin_targets


class ConvolutionALNet(nn.Module):
    def __init__(
        self, backbone: nn.Module, output_size: int, backbone_output_size: int = 0
    ) -> None:
        super(ConvolutionALNet, self).__init__()
        self._backbone = backbone
        self._output_size = output_size
        self._inited = False

        # for a classificaiton problem, the following members hold the binary target values
        self._bin_targets = None
        self._inputs_targets = None

        last_module = list(backbone.modules())[-1]
        if not hasattr(last_module, "out_features") and backbone_output_size == 0:
            raise ValueError(
                "Cannot get the output size of the backbone module. "
                "Pass the output size in backbone_output_size"
            )

        if backbone_output_size != 0:
            self._backbone_out_size = backbone_output_size
        elif hasattr(last_module, "out_features"):
            self._backbone_out_size = last_module.out_features

    def init_alns(self, **kwargs):
        """
        Initializes ALNs as the network outputs.

        Args:
            init_pieces: (int) Number of initial LUs
            root_op: (str) "min", "max", "random". Default is 'random'.
            ops: (str), "min", "max", "random" or "alt. Default is 'random'.
                Other ops in the tree, if init_pieces > 1. 'alt' stands for alternate.
            device: Device to init and run the model on.
            dtype: Parameters data type

          (Only pass the following if you know what you are doing)
            epsilons: float
            min_constraints: float
            max_constraints: float
        """
        for i in range(self._output_size):
            # set the alns as attributes so their parameters automatically register
            setattr(
                self,
                f"aln{i}",
                ALNet.create_full_tree(
                    input_features=self._backbone_out_size,
                    layer_count=kwargs.get("init_pieces", 1),
                    root_op=kwargs.get("root_op", "random"),
                    ops=kwargs.get("ops", "alt"),
                    dtype=kwargs.get("dtype", None),
                    device=kwargs.get("device", None),
                    epsilons=kwargs.get("epsilons", None),
                    min_constraints=kwargs.get("min_constraints", None),
                    max_constraints=kwargs.get("max_constraints", None),
                ),
            )

    def forward(self, x):
        x = self._backbone(x)
        outputs = []
        for i in range(self._output_size):
            outputs.append(getattr(self, f"aln{i}")(x))
        outputs = torch.stack(outputs, dim=1)
        return outputs.squeeze(-1)

    def adapt(self, x, output, targets, lr):
        x = self._backbone(x)
        for i in range(self._output_size):
            aln = getattr(self, f"aln{i}")
            aln.adapt(x, output[:, i], targets[:, i], lr)

    def grow(self, loader, min_rmse, max_splits):
        self.eval()

        # The first time we call grow, we need to convert the target to binary
        if self._bin_targets is None:
            self._bin_targets = {}
            self._inputs_targets = []
            for i in range(self._output_size):
                self._bin_targets[i] = []

            for inputs, target in loader:
                self._inputs_targets.append(inputs)
                target = get_bin_targets(target, self._output_size, -1, as_dict=True)
                for k, v in target.items():
                    self._bin_targets[k].append(v)

        for i in range(self._output_size):
            aln = getattr(self, f"aln{i}")
            grow_loader = list(zip(self._inputs_targets, self._bin_targets[i]))
            random.shuffle(grow_loader)
            aln.grow(
                grow_loader,
                min_rmse=min_rmse,
                preprocess=self._backbone,
                max_split_count=max_splits,
            )
        self.train()
