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

import os
import sys
import json
import pickle
import numpy as np
from copy import deepcopy
from datetime import datetime

import itertools
import matplotlib.pyplot as plt
from itertools import product

from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable

WANDB_ENTITY_NAME = "asg-ml-team"


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    NORM = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_title(msg):
    msg = "+++ " + msg + " +++"
    amsg = bcolors.OKBLUE + "\n{0}\n" + msg + "\n{0}" + bcolors.NORM
    amsg = amsg.format("=" * len(msg))
    if amsg.find("\t") > -1:
        amsg = amsg.expandtabs(4)
    print(amsg)


def print_error(msg, timestamp=True, CR=True):
    if CR:
        msg = msg + "\n"

    if timestamp:
        msg = f"[{str(datetime.now())}] {msg}"

    msg = bcolors.FAIL + msg + bcolors.NORM

    if msg.find("\t") > -1:
        msg = msg.expandtabs(4)
    print(msg)


def print_warning(msg, timestamp=True, CR=True):
    if CR:
        msg = msg + "\n"

    if timestamp:
        msg = f"[{str(datetime.now())}] {msg}"

    msg = bcolors.WARNING + msg + bcolors.NORM
    if msg.find("\t") > -1:
        msg = msg.expandtabs(4)
    print(msg)


def print_info(msg, timestamp=True, CR=True):
    if CR:
        msg = msg + "\n"
    if timestamp:
        msg = f"[{str(datetime.now())}] {msg}"
    msg = bcolors.BOLD + msg + bcolors.NORM
    if msg.find("\t") > -1:
        msg = msg.expandtabs(4)
    print(msg)


def get_current_dir():
    return os.path.dirname(sys.argv[0])


def count_subfolders(folder):
    if not os.path.isdir(folder):
        return 0
    return len(os.listdir(folder))


def subdirs(path):
    for entry in os.scandir(path):
        if not entry.name.startswith(".") and entry.is_dir():
            yield entry.name


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def store_json(data, filename):
    with open(filename, "w") as f:
        return json.dump(data, f)


def store_lines(lines, filename, encoding="utf-8"):
    with open(filename, "w", encoding=encoding) as f:
        lines = [l.strip() + os.linesep for l in lines]
        f.writelines(lines)


def load_pickle(filename):
    with open(filename, mode="rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def count_parameters(model, additional_params=0, no_show=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if not no_show:
        print(table)
        print(f"Total Trainable Params: {total_params}")
        if additional_params > 0:
            print(f"Additional Params: {additional_params}")
            print(f"All params: {total_params+additional_params}")

    return total_params + additional_params


def plot_confusion_matrix(
    y_true, y_pred, labels=None, normalize=False, model_name=None, figsize=None
):
    """
    Generate matrix plots of confusion matrix for multi-class classification.

    Inputs:
      y_true: true label of the data, with shape (nsamples, nclasses)
      y_pred: prediction of the data, with shape (nsamples, nclasses)
      labels: string array, name the order of class labels in the confusion matrix.
      normalize: normalize the values to be in [0,1].
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true shape {y_true.shape} does not match y_pred {y_pred.shape}"
        )

    n_classes = len(np.unique(y_true))

    if labels is None:
        labels = range(n_classes)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        cm = cm.astype(int)
        fmt = "d"

    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    plt.colorbar()

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > thresh else "black"
        text = format(cm[i, j], fmt)
        plt.text(j, i, text, horizontalalignment="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if model_name:
        plt.title(f"Confusion Matrix for {model_name}")
    plt.margins(y=0)

    return fig
