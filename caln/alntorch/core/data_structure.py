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

from collections import namedtuple

import torch
import torch.nn as nn
import torch.autograd as autograd


def validate_node_type(node):
    if node is None or (
        not isinstance(node, Linear)
        and not isinstance(node, Max)
        and not isinstance(node, Min)
    ):
        raise ValueError("node must be an instance of Linear, Max, or Min")


class LinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        ctx.save_for_backward(weight)
        output = inputs.mm(weight.t())
        output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        # we use a custom routine for weight and bias updates,
        # so do not return any gradient for them
        return grad_input, None, None


WeightConstraint = namedtuple("WeightConstraint", ["min", "max"])


class Linear(nn.Module):
    def __init__(
        self,
        node_id: int,
        input_features: int,
        epsilons: torch.Tensor,
        min_constraints: torch.Tensor,
        max_constraints: torch.Tensor,
        dtype=None,
        device=None,
    ):

        super(Linear, self).__init__()

        assert epsilons.size(0) == input_features
        assert min_constraints.size(0) == input_features
        assert max_constraints.size(0) == input_features

        self.id = node_id
        self.input_features = input_features
        self.min_constraints = min_constraints
        self.max_constraints = max_constraints
        self.dtype = dtype
        self.device = device

        self._is_constant = False
        self._is_split_allowed = True

        self.register_buffer("epsilons", epsilons)

        # weight
        self.register_buffer(
            "weight",
            torch.rand(
                1, self.input_features, dtype=dtype, device=device, requires_grad=False
            ),
        )

        # bias
        self.register_buffer(
            "bias", torch.rand(1, dtype=dtype, device=device, requires_grad=False)
        )

        # centroid (mean) over all input points
        self.register_buffer(
            "m_input", torch.zeros_like(self.weight.data, requires_grad=False)
        )
        self.register_buffer(
            "m_output", torch.zeros_like(self.bias.data, requires_grad=False)
        )

        # variance
        self.register_buffer(
            "v_input", torch.zeros_like(self.weight.data, requires_grad=False)
        )
        self.v_input += torch.square(self.epsilons)

        # init id column
        self.register_buffer(
            "_id_col",
            torch.zeros(1, 1, dtype=torch.int32, device=device, requires_grad=False),
        )
        self._id_col.fill_(self.id)

    @property
    def is_constant(self):
        return self._is_constant

    @is_constant.setter
    def is_constant(self, value):
        self._is_constant = value
        if self._is_constant:
            self._is_split_allowed = False

    @property
    def is_split_allowed(self):
        return self._is_split_allowed

    @is_split_allowed.setter
    def is_split_allowed(self, value):
        if value and self.is_constant:
            raise ValueError("A constant piece cannot be allowed to split")

        self._is_split_allowed = value

    @torch.no_grad()
    def set_constant_value(self, value, d=0.0000001):
        if d <= 0:
            raise ValueError("d must be positive")

        self.is_constant = True
        assert not self.is_split_allowed

        # all training variance and centroid tensors reset to zero
        self.v_input.fill_(d)
        self.m_input.fill_(0)
        self.m_output.fill_(0)

        # weights are now zero
        self.weight.fill_(0)

        # and bias is set to the constant value
        self.bias.fill_(value)

    @torch.no_grad()
    def adapt(self, inputs, actual, target, responsibilities, lr):

        # responsibilities.sum() is the number of rows this pieces is responsible for
        count = responsibilities.sum()
        if count == 0:
            return
        lr = lr / count

        # distribute adaptation over n weight and n + 1 centroid elements
        distributed_lr = lr / (
            self.input_features * 2 + 1
        )  # mothomas: do we need to worry about this getting too small?

        error = (responsibilities * (actual - target)).sum() / count

        # even if the piece is constant, we still update variance and centroid
        # as useful statistics; in this case, the centroid will no longer lie
        # on the line

        # distance from point to centroid in input and out dimensions
        d_input = (responsibilities * (inputs - self.m_input)).sum() / count
        d_output = (responsibilities * (target - self.m_output)).sum() / count

        # update the variance of distance of points to the centroid
        self.v_input += (torch.square(d_input) - self.v_input) * lr
        self.v_input.clamp_(min=0.0000001)

        # update the centroid (mean)
        self.m_input += d_input * distributed_lr
        self.m_output += d_output * distributed_lr

        # update weights and bias only if not constant
        if not self.is_constant:

            # The closer a point is to the centroid, the less change in the weights.
            # That is, points at the centroid will have the least effect on the weight,
            # (since d_input is zero)... so for points near the centroid, adaptation
            # to the target will be achieved more by changing the bias than by changing
            # the weights.
            self.weight -= (error * d_input / self.v_input) * distributed_lr

            # not sure if cliping weights is healthy for training
            # self.weight.clamp_(self.min_constraints, self.max_constraints)

            # recalculate bias from updated weights and centroid
            self.bias.data = self.m_output + self.m_input.mm(-self.weight.t())[0]

    @torch.no_grad()
    def copy(self, copy_id):
        other = Linear(
            copy_id,
            self.input_features,
            epsilons=self.epsilons,
            min_constraints=self.min_constraints,
            max_constraints=self.max_constraints,
            dtype=self.dtype,
            device=self.weight.device,
        )

        other._is_split_allowed = self._is_split_allowed
        other._is_constant = self._is_constant

        other.weight.copy_(self.weight.data)
        other.bias.copy_(self.bias.data)
        other.m_input.copy_(self.m_input.data)
        other.m_output.copy_(self.m_output.data)
        other.v_input.copy_(self.v_input.data)
        return other

    def forward(self, inputs):
        result = LinearFunction.apply(inputs, self.weight, self.bias)

        if self._id_col.shape != result.shape:
            self._id_col.resize_(result.shape)
            self._id_col.fill_(self.id)

        return result, self._id_col

    def __repr__(self):
        return (
            f"Linear(id:{self.id})"
            if not self.is_constant
            else f"Constant(id:{self.id})"
        )


class Operator(nn.Module):
    def __init__(self, left, right, is_min):
        validate_node_type(left)
        validate_node_type(right)

        super(Operator, self).__init__()

        self.is_min = is_min
        self.left = left
        self.right = right

    def forward(self, input):
        l_result, l_id = self.left(input)
        r_result, r_id = self.right(input)

        result = (
            torch.minimum(l_result, r_result)
            if self.is_min
            else torch.maximum(l_result, r_result)
        )

        # piece id selector
        # where value is
        #   true if left id should be selected
        #   false if right id should be selected
        id_selector = (
            l_result < r_result  # min, take left if smaller
            if self.is_min
            else l_result > r_result  # max, take left if larger
        )

        # convert id selector to rows for use with torch.where
        id_selector = id_selector.reshape(l_result.size(0), 1)
        ids = torch.where(id_selector, l_id, r_id)
        return result, ids

    def load_state_dict(self, state_dict, strict=True):
        super(Operator, self).load_state_dict(state_dict, strict=strict)
        self.last_responsible_pieces.clear()


class Max(Operator):
    def __init__(self, left, right):
        super(Max, self).__init__(left, right, is_min=False)


class Min(Operator):
    def __init__(self, left, right):
        super(Min, self).__init__(left, right, is_min=True)
