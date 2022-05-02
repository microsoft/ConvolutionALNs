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
import enum

from typing import Callable, Union
from collections import namedtuple

import torch
import torch.nn as nn

from .data_structure import Linear, Min, Max


class SplitPreference(enum.Enum):
    DATA_DRIVEN = 0
    MIN = 1
    MAX = 2


AlnNodeArgs = Union[Linear, Min, Max]
MinMaxNodeArgs = Union[Min, Max]


class ALNet(nn.Module):

    _SplitCandidate = namedtuple(
        "_SplitCandidate", ["piece", "parent", "count", "rmse", "t"]
    )
    _Responsibility = namedtuple("_Responsibility", ["inputs", "target", "error"])
    _CentroidStats = namedtuple(
        "_CentroidStats", ["piece", "inputs", "target", "count"]
    )

    @staticmethod
    def create_full_tree(
        input_features,
        layer_count: int,
        root_op: str = "random",  # can be "min", "max" or "random"
        ops: str = "alt",  # can be "min", "max", "random" or "alt" as alternate
        epsilons=None,
        min_constraints=None,
        max_constraints=None,
        dtype=None,
        device=None,
    ):

        if layer_count <= 0:
            raise ValueError("layer_count must be greater than 0")

        aln = ALNet(
            input_features,
            epsilons=epsilons,
            min_constraints=min_constraints,
            max_constraints=max_constraints,
            dtype=dtype,
            device=device,
        )

        if layer_count == 1:
            return aln

        ops = ops.lower()
        root_op = root_op.lower()

        if ops not in ["min", "max", "random", "alt"]:
            raise ValueError(
                f"Invalid value for ops, '{ops}'. "
                "Can be eiter 'max', 'min', 'random' or 'alt'."
            )

        if root_op not in ["min", "max", "random"]:
            raise ValueError(
                f"Invalid value for root_op, '{root_op}'. "
                "Can be eiter 'max', 'min' or 'random'."
            )

        pieces_and_parents = []

        def gather_pieces_and_parents(parent, child):
            if isinstance(child, Linear):
                pieces_and_parents.append((child, parent))

        if root_op == "min":
            is_min = True
        elif root_op == "max":
            is_min = False
        elif root_op == "random":
            is_min = random.randint(0, 1) == 0

        while layer_count > 1:
            pieces_and_parents.clear()
            aln.visit_nodes(gather_pieces_and_parents)
            split_preference = SplitPreference.MIN if is_min else SplitPreference.MAX
            for piece, parent in pieces_and_parents:
                aln._split(piece, parent, split_preference)

            if ops == "alt":
                is_min = not is_min  # switch node type for next layer
            elif ops == "max":
                is_min = False
            elif ops == "min":
                is_min = True
            elif ops == "random":
                is_min = random.randint(0, 1) == 0

            layer_count -= 1

        return aln

    @staticmethod
    def create_bounded_output_tree(
        input_features,
        max_output_bound,
        min_output_bound,
        epsilons=None,
        min_constraints=None,
        max_constraints=None,
        dtype=None,
        device=None,
    ):

        #
        # We want a network of three linear pieces:
        #
        # min
        # min.right,      <---- linear, will be constant lower bound
        # max = min.left
        # max.left        <---- linear
        # max.right       <---- linear, will be constant upper bound
        #
        # we constrain network output to be within [min_output_bound, max_output_bound]:
        #   - min.right is constant to cutoff any values above max_output_bound
        #   - max.right is constant to cutoff any values below min_output_bound
        #   - max.left can adapt and be grown
        #

        # create a min tree of two linear elements
        aln = ALNet.create_full_tree(
            input_features,
            2,
            root_is_min=True,
            epsilons=epsilons,
            min_constraints=min_constraints,
            max_constraints=max_constraints,
            dtype=dtype,
            device=device,
        )
        min_node = aln.root

        # split left child into a max
        max_node = aln._split(min_node.left, min_node, SplitPreference.MAX)

        # set two pieces constant
        min_node.right.set_constant_value(max_output_bound)
        max_node.right.set_constant_value(min_output_bound)

        assert min_node.right.is_constant
        assert max_node.right.is_constant

        # third piece can grow and adapt
        assert max_node.left.is_split_allowed and not max_node.left.is_constant

        return aln

    def __init__(
        self,
        input_features,
        epsilons=None,
        min_constraints=None,
        max_constraints=None,
        dtype=None,
        device=None,
    ):
        super(ALNet, self).__init__()
        self.input_features = input_features
        self.dtype = dtype
        self.device = device
        self.pieces = {}

        if dtype is None:
            dtype = torch.get_default_dtype()

        if epsilons is None:
            epsilons = torch.zeros(input_features, dtype=dtype, device=device)
            epsilons += 0.001

        if min_constraints is None:
            min_constraints = torch.zeros(input_features, dtype=dtype, device=device)
            min_constraint_value = torch.finfo(min_constraints.dtype).min
            min_constraints.fill_(min_constraint_value)

        if max_constraints is None:
            max_constraints = torch.zeros(input_features, dtype=dtype, device=device)
            max_constraint_value = torch.finfo(max_constraints.dtype).max
            max_constraints.fill_(max_constraint_value)

        assert isinstance(epsilons, torch.Tensor) and epsilons.size(0) == input_features
        assert (
            isinstance(min_constraints, torch.Tensor)
            and min_constraints.size(0) == input_features
        )
        assert (
            isinstance(max_constraints, torch.Tensor)
            and max_constraints.size(0) == input_features
        )

        self.register_buffer("epsilons", epsilons)
        self.register_buffer("min_constraints", min_constraints)
        self.register_buffer("max_constraints", max_constraints)
        self.register_buffer(
            "responsible_ones",
            torch.ones(1, 1, dtype=dtype, device=device, requires_grad=False),
        )
        self.register_buffer(
            "responsible_zeros",
            torch.zeros(1, 1, dtype=dtype, device=device, requires_grad=False),
        )

        self.root = Linear(
            0,
            input_features,
            epsilons=self.epsilons,
            max_constraints=self.max_constraints,
            min_constraints=self.min_constraints,
            dtype=dtype,
            device=device,
        )

        self.pieces[self.root.id] = self.root
        self.responsible_pieces = None

    def to(self, device):
        self.device = device
        return super().to(device)

    def visit_nodes(self, visitor: Callable[[AlnNodeArgs, AlnNodeArgs], None]):
        stack = [(None, self.root)]
        while len(stack) > 0:
            parent, child = stack.pop()
            visitor(parent, child)
            if not isinstance(child, Linear):
                stack.append((child, child.left))
                stack.append((child, child.right))

    def forward(self, inputs):
        result, responsible_pieces = self.root(inputs)
        self._responsible_pieces = responsible_pieces
        return result

    @torch.no_grad()
    def adapt(self, inputs, actual, target, lr):
        # make sure the ones and zeroes vectors are the right size
        batch_size = inputs.size(0)
        if batch_size != self.responsible_ones.size(0):
            self.responsible_ones.resize_(batch_size, 1)
            self.responsible_ones.fill_(1)
            self.responsible_zeros.resize_(batch_size, 1)
            self.responsible_zeros.fill_(0)

        # convert to rows for use with torch.where
        responsible_pieces = self._responsible_pieces.reshape(batch_size, 1)
        unique_pieces = torch.unique(
            responsible_pieces,
        )
        for piece_id in unique_pieces:
            piece_id = piece_id.item()
            # responsibilities is a vector of 1s and 0s.
            # If a piece is responsible for an input, the corresponding row will be 1, otherwise 0.
            responsibilities = torch.where(
                responsible_pieces == piece_id,
                self.responsible_ones,
                self.responsible_zeros,
            )
            piece = self.pieces[int(piece_id)]
            piece.adapt(inputs, actual, target, responsibilities, lr)

    def validate_compatible_linear_piece(self, node: Linear) -> bool:
        if not node.input_features == self.input_features:
            raise ValueError(
                f"Linear node input_features {node.input_features} "
                f"must equal aln input_features {self.input_features}"
            )
        if not node.dtype == self.dtype:
            raise ValueError(
                f"Linear node dtype {node.dtype} must equal aln dtype {self.dtype}"
            )
        if not node.device == self.device:
            raise ValueError(
                f"Linear node device {node.device} must equal aln device {self.device}"
            )

    def _split(self, piece, parent, split_preference: SplitPreference, t=None):

        if not piece.is_split_allowed:
            raise ValueError(f"Splitting the linear piece id={piece.id} is not allowed")

        # determine new parent type
        old_parent = parent
        if split_preference == SplitPreference.DATA_DRIVEN:
            assert t is not None
            if t == 0:
                # choose same as parent, or MIN if parent is NONE
                split_preference = (
                    SplitPreference.MAX
                    if isinstance(old_parent, Max)
                    else SplitPreference.MIN
                )
            else:
                # choose MAX if t > 0 else MIN
                split_preference = SplitPreference.MAX if t > 0 else SplitPreference.MIN

        # remove piece from existing parent
        replace_left_child = None
        if old_parent is not None:
            replace_left_child = True if piece == old_parent.left else False
            if replace_left_child:
                del old_parent.left
            else:
                del old_parent.right

        # create copy of left as new piece
        left = piece

        next_piece_id = len(self.pieces)
        right = piece.copy(next_piece_id)

        self.pieces[right.id] = right

        # create new parent
        new_parent = (
            Max(left, right)
            if split_preference == SplitPreference.MAX
            else Min(left, right)
        )

        # place new parent in old parent or as root
        if old_parent is None:
            del self.root
            self.root = new_parent
        else:
            if replace_left_child:
                old_parent.left = new_parent
            else:
                old_parent.right = new_parent

        return new_parent

    @torch.no_grad()
    def grow(
        self,
        loader,
        max_split_count=1,  # by default we only split 1 LU, < 0 for all
        min_rmse=0,
        min_responsible_points=0,
        split_preference=SplitPreference.DATA_DRIVEN,
        output_op=None,
        preprocess=None,
    ) -> int:

        # assign piece reponsibility, keyed by linear piece id, for each data point in the loader
        responsibility_by_piece_id = {}
        for inputs, target in loader:
            if self.device != "cpu":
                inputs = inputs.to(self.device)
                if preprocess is not None:
                    inputs = preprocess(inputs)
                target = target.to(self.device)
            output, responsible_pieces = self.root(inputs)
            if output_op is not None:
                output = output_op(output)
            error = (
                output.squeeze(-1) - target
                if len(target.shape) == 1
                else output - target
            )

            # convert tensor containing piece ids to list
            responsible_pieces = responsible_pieces.flatten().tolist()
            for i, piece_id in enumerate(responsible_pieces):
                stats = responsibility_by_piece_id.get(piece_id)
                if stats is None:
                    stats = []
                    responsibility_by_piece_id[piece_id] = stats

                stats.append(
                    ALNet._Responsibility(
                        inputs=inputs[i, :], target=target[i], error=error[i]
                    )
                )

        # gather parent/child relationship of all linear pieces, keyed by linear piece id
        parent_child_by_linear_id = {}

        def gather_linear_parents_visitor(parent, child):
            if isinstance(child, Linear):
                parent_child_by_linear_id[child.id] = (parent, child)

        self.visit_nodes(gather_linear_parents_visitor)

        # build list of split candidates based on distribution of points each piece is responsible for
        candidates = []
        for key, stats in responsibility_by_piece_id.items():

            parent, piece = parent_child_by_linear_id[key]
            if not piece.is_split_allowed:
                continue

            # calc sse, mean and variance of inputs / targets (one pass)
            # see https://www.johndcook.com/blog/standard_deviation/
            count = len(stats)
            if count < min_responsible_points:
                continue

            stat0 = stats[0]
            sse = torch.zeros_like(stat0.error)
            mean = torch.zeros_like(stat0.inputs)
            variance = torch.zeros_like(stat0.inputs)
            if count == 1:
                mean += stat0.inputs
                sse += torch.square(stat0.error)
            else:
                for i, (inputs, _, error) in enumerate(stats):
                    sse += torch.square(error)
                    diff = inputs - mean
                    mean += diff / (i + 1)
                    variance += diff * (inputs - mean)
                variance /= count - 1

            rmse = sse.sqrt() / count
            if rmse < min_rmse:
                continue

            # calc the distribution of input points around the centroid (mean)
            b = torch.zeros_like(mean)
            for inputs, _, error in stats:
                diff = inputs - mean
                diff_variance = variance - (diff * diff)
                b += diff_variance * error

            # sum the bend components;
            # if > 0 then points are distributed in a convex manner around the piece
            # if < 0 then points are distributed in a concave manner around the piece
            t = torch.sum(b)

            candidates.append(
                ALNet._SplitCandidate(
                    piece=piece, parent=parent, count=count, rmse=rmse, t=t
                )
            )

        if len(candidates) == 0:
            return 0

        # sort candidates by rmse in descending order
        candidates.sort(key=lambda c: c.rmse, reverse=True)

        # perform splits
        if max_split_count < 0:
            split_count = len(candidates)
        else:
            split_count = min(len(candidates), max_split_count)
        for candidate in candidates[:split_count]:
            self._split(
                candidate.piece, candidate.parent, split_preference, candidate.t
            )

        return split_count
