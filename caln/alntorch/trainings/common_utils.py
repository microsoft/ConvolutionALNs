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

import numpy as np
import argparse

import torch


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train ALN variations on CIFAR10",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of this run. Used for monitoring and checkpointing.",
    )
    parser.add_argument(
        "--logdir",
        default="./",
        help="Root path to log training info.",
    )
    parser.add_argument(
        "--cifar10_path",
        default="./",
        help="Path to store CIFAR-10 data.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Maximum number of epochs.",
        default=200,
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--aln_lr", type=float, default=1, help="ALN Learning rate.")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Train batch size.",
        default=256,
    )
    parser.add_argument(
        "--early_stopping", type=bool, default=False, help="Use early stopping."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device.",
        choices=["cpu"] + [f"cuda:{i}" for i in range(8)],
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of workers for dataloader.",
    )

    parser.add_argument(
        "--loss",
        choices=["CE", "MSE"],
        default="CE",
        help="Train loss.",
    )

    parser.add_argument(
        "--optimizer",
        choices=["AdamW", "SGD"],
        default="AdamW",
        help="Optimizer.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_annealing",
        help="Should type.",
    )

    parser.add_argument(
        "--model",
        choices=[
            "CALN",
            "ResNet14",
            "ResNet18",
        ],
        default="CALN",
        help="Choose the model variant",
    )
    parser.add_argument(
        "--split_cap",
        type=int,
        help="Maximum number of split iterations. -1 for infinite.",
        default=-1,
    )
    parser.add_argument(
        "--split_step",
        type=int,
        default=1,
        help="Step (iteration number) to perform Split.",
    )
    parser.add_argument(
        "--split_step_increment",
        type=int,
        help="Number of step increments for split step after each split.",
        default=0,
    )

    parser.add_argument(
        "--max_splits",
        type=int,
        help="Maximum number of LUs to split at each split iteration. "
        + "Negative value causes all LUs to split at each split iteration.",
        default=0,
    )

    parser.add_argument(
        "--init_pieces", type=int, default=1, help="Initial pieces count."
    )
    parser.add_argument(
        "--root_op",
        help="In case of init_pieces > 1, determine the root op (min or max or random)",
        default="random",
        choices=["random", "min", "max"],
    )

    parser.add_argument(
        "--backbone_id",
        type=int,
        help="Backbone configuration ID.",
        choices=[0, 1, 2, 3, 4, 5],
        default=0,
    )

    parser.add_argument(
        "--logger",
        type=str,
        help="Train logger. ",
        choices=["wandb", "tensorboard"],
        default="tensorboard",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Project name for wandb logger. ",
        default="CALN",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        help="Entity name for wandb logger ",
        default="",
    )

    return parser


def py_class_value_to_binary(aclass, value, min_value=-1.0, max_value=1.0):
    return max_value if abs(aclass - value) < 0.5 else min_value


class_value_to_binary = np.vectorize(py_class_value_to_binary, otypes=[np.float32])


def get_bin_targets(
    targets, n_classes, min_target_val=-1, max_target_val=1, device="cpu", as_dict=False
):
    bin_targets = {}
    for i in range(n_classes):
        aux = class_value_to_binary(targets, i, min_target_val, max_target_val)
        bin_targets[i] = torch.tensor(aux, device=device)
    if not as_dict:
        bin_targets = torch.stack([bin_targets[i] for i in range(n_classes)], dim=-1)

    return bin_targets
