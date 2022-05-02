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

from alntorch.core.caln import ConvolutionALNet

from alntorch.trainings.common_utils import get_bin_targets
from alntorch.trainings.resnet import (
    MResNet,
    BasicBlock,
    Bottleneck,
    ResNet14,
    ResNet18,
)


def get_model(args):
    if args.model == "ResNet14":
        return ResNet14()
    elif args.model == "ResNet18":
        return ResNet18()
    elif args.model == "CALN":
        return ALNVariants(**vars(args))
    else:
        raise ValueError(f"Unrecognized model {args.model}")


class ALNVariants(ConvolutionALNet):
    def __init__(self, backbone_id=None, nclasses=10, **kwargs):
        device = kwargs.get("device", "cpu")

        if backbone_id is None:
            block = BasicBlock

        if backbone_id == 0:  # ResNet13
            layers_and_planes = [[2, 64], [2, 128], [2, 256]]
            self._aln_input_size = 1024
            block = BasicBlock

        elif backbone_id == 1:
            layers_and_planes = [[3, 64], [4, 128], [10, 256]]
            block = Bottleneck
            self._aln_input_size = 1024 * block.expansion

        elif backbone_id == 2:
            layers_and_planes = [[3, 64], [4, 128], [23, 256], [3, 512]]
            block = Bottleneck
            self._aln_input_size = 512 * block.expansion

        elif backbone_id == 3:
            layers_and_planes = [[2, 64], [2, 128], [2, 256], [2, 512]]
            self._aln_input_size = 512
            block = BasicBlock

        elif backbone_id == 4:
            layers_and_planes = [[2, 64], [2, 128]]
            self._aln_input_size = 2048
            block = BasicBlock
        else:
            raise NotImplementedError("backbone {backbon_id} not recognized.")

        backbone = MResNet(block, layers_and_planes)
        backbone = backbone.to(device)
        super(ALNVariants, self).__init__(
            backbone, output_size=nclasses, backbone_output_size=self._aln_input_size
        )
        self.init_alns(**kwargs)

    def count_lu_params(self):
        lu_params = 0
        for i in range(self._output_size):
            aln = getattr(self, f"aln{i}")
            lu_params += len(aln.pieces) * self._aln_input_size
        return lu_params
