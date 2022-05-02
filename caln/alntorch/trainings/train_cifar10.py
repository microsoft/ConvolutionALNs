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
import time
import datetime
import numpy as np


import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score

import alntorch.core.utils as utils
from alntorch.trainings.models import get_model
from alntorch.trainings.train_logger import TrainLogger
from alntorch.trainings.common_utils import argparser
from alntorch.trainings.common_utils import get_bin_targets

os.environ["WANDB_SILENT"] = "true"


def get_cifar10_loaders(path, train_batchsize, test_batchsize, num_workers=10):
    train_tx = tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    val_tx = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = tv.datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=train_tx,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batchsize, shuffle=False, num_workers=num_workers
    )

    testset = tv.datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=val_tx,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batchsize, shuffle=False, num_workers=num_workers
    )

    class_names = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return train_loader, test_loader, class_names


def get_optimizer(optimizer_type):
    if optimizer_type == "AdamW":
        return optim.AdamW(network.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        return optim.SGD(
            network.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
    else:
        raise ValueError(
            f"Invalid or unsupported optimizer {optimizer_type}."
            "Supported optimizers are AdamW and SGD."
        )


def get_scheduler(scheduler_type=None, optimizer=None, n_epochs=0):
    if scheduler_type:
        if optimizer is None or n_epochs < 1:
            raise ValueError("Invalid parameters for scheduler.")

        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif scheduler_type is None:
        return None
    else:
        raise ValueError(f"Invalid or unsupported scheduler {scheduler_type}.")


def get_criterion(loss):
    loss = loss.lower()
    if loss == "ce":
        return nn.CrossEntropyLoss()
    elif loss == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Invalid or unsupported loss {loss}.")


def test(network, test_loader, criterion, device="cpu"):
    network.eval()
    test_loss = 0
    y_pred = []
    y_true = []
    test_times = []

    # for the zip generator lambda function
    if callable(test_loader):
        test_loader = test_loader()

    with torch.no_grad():
        for data, target in test_loader:
            if len(target.shape) > 1:
                y_true.extend(torch.argmax(target, dim=-1))
            else:
                y_true.extend(target)
            data = data.to(device)
            target = target.to(device)

            t0 = time.time()
            output = network(data).float()
            t1 = time.time()

            test_times.append(t1 - t0)

            test_loss += criterion(output, target).item()
            output = (
                torch.argmax(output, dim=1)
                if device == "cpu"
                else torch.argmax(output, dim=1).cpu()
            )

            y_pred.extend(output)

    test_loss /= len(y_true)
    test_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    test_macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics = {
        "loss": test_loss,
        "y_pred": y_pred,
        "y_true": y_true,
        "accuracy": test_accuracy,
        "macro_precision": test_macro_precision,
        "avg_time": np.mean(test_times),
    }
    utils.print_info(
        f"> Evaluation results: "
        f" Accuracy = {test_accuracy}, "
        f"Loss = {test_loss:0.5}, "
        f"Average time = {metrics['avg_time']:0.4} seconds."
    )

    return metrics


def report_and_log(metrics, class_names, epoch, logger):
    report = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    avg_pre = 0
    avg_rec = 0
    avg_f1 = 0
    for c in class_names:
        info = {
            "Precision": report[c]["precision"],
            "Recall": report[c]["recall"],
            "F1-Score": report[c]["f1-score"],
            "Step": epoch,
        }
        logger.log({f"Test/{c}/": info}, step=epoch)
        avg_pre += report[c]["precision"]
        avg_rec += report[c]["recall"]
        avg_f1 += report[c]["f1-score"]

    logger.log(
        {
            "Test/MultiClass/": {
                "Accuracy": report["accuracy"],
                "Loss": metrics["loss"],
                "Step": epoch,
            }
        },
        step=epoch,
    )
    logger.log_tensorboard_additional(metrics, class_names, step=epoch, tag="test")
    return report


if __name__ == "__main__":
    args = argparser().parse_args()

    utils.print_title(f"CIFAR10 training - Name: {args.name}")

    # The default experiment name is test (case insensitive), note that
    # an already existing folder named test will be overwritten. If you
    # want not to overwrite past experiments use another name.
    experiment_name = args.name
    args.logdir = os.path.join(args.logdir, args.name)

    os.makedirs(args.logdir, exist_ok=True if args.name.upper() == "TEST" else False)
    logger = TrainLogger(args.logger, **vars(args))

    n_epochs = args.epochs
    learning_rate = args.lr
    aln_learning_rate = args.aln_lr
    batch_size = args.batch_size
    early_stopping = args.early_stopping

    device = args.device
    num_workers = args.num_workers

    if device != "cpu":
        torch.backends.cudnn.enabled = True

    split_cap = args.split_cap
    split_step = args.split_step
    split_incr = args.split_step_increment
    max_splits = args.max_splits
    loss_type = args.loss
    optimizer_type = args.optimizer
    scheduler_type = args.scheduler

    # It is possible to use this code for training other than ALN-based networks (refer to the docs)
    is_aln = True if args.model == "CALN" else False

    train_loader, test_loader, class_names = get_cifar10_loaders(
        path=args.cifar10_path,
        train_batchsize=batch_size,
        test_batchsize=500,
        num_workers=num_workers,
    )

    n_classes = len(class_names)
    network = get_model(args)
    network = network.to(device)

    criterion = get_criterion(loss_type)
    optimizer = get_optimizer(optimizer_type)
    scheduler = get_scheduler(scheduler_type, optimizer, n_epochs)

    if is_aln:
        args.nparams = utils.count_parameters(
            network, additional_params=network.count_lu_params()
        )
    else:
        args.nparams = utils.count_parameters(network)

    # convert to binary targets if MSE loss (only for test because of augmentation in train)
    if loss_type == "MSE":
        # We need binary targets in case the loss is MSE
        test_inputs = []
        test_targets = []
        for inputs, target in test_loader:
            test_inputs.append(inputs)
            test_targets.append(
                get_bin_targets(target, n_classes, min_target_val=-1, as_dict=False)
            )
        test_loader = lambda: zip(test_inputs, test_targets)

    # initial logging and report
    metrics = test(network, test_loader, criterion, device)
    report = report_and_log(metrics, class_names, 0, logger)

    last_acc = 0
    lr_apply_step = 0
    split_counter = 0
    best_accuracy = -np.inf
    n_early_stopping = 0

    t0_train = time.time()
    # Main training loop
    for epoch in range(1, n_epochs + 1):

        # Split
        if is_aln and epoch % split_step == 0:
            if split_cap > -1 and split_counter > split_cap:
                utils.print_info(
                    "Maximum split steps reached. Network will no longer grow."
                )
            else:
                split_counter += 1

                if split_counter == 1:
                    split_increase = split_step + split_incr
                else:
                    split_increase += split_incr

                split_step += split_increase

                utils.print_info(
                    f"Growing the network ({split_counter}), this may take a while. Next split at ({split_step}). ",
                    CR=False,
                )
                prev_params = network.count_lu_params()
                network.grow(train_loader, min_rmse=0.00, max_splits=max_splits)
                new_params = network.count_lu_params()

                utils.print_info(
                    f"Growing successfull. Param count increases from {prev_params} to {new_params}. "
                )

        network.train()
        utils.print_info(f"Epoch #{epoch}", CR=False)

        # Fit
        bin_targets = {}
        t0 = time.time()
        for data, target in train_loader:
            optimizer.zero_grad()

            data = data.to(device).float()
            output = network(data).float()

            bin_target = get_bin_targets(
                target,
                n_classes,
                device=device,
                min_target_val=-1,
                as_dict=False,
            )
            if loss_type == "CE":
                target = target.to(device)
                loss = criterion(output, target)
            else:
                loss = criterion(output, bin_target)

            loss.backward()
            optimizer.step()

            if is_aln:
                network.adapt(
                    data,
                    output,
                    bin_target,
                    aln_learning_rate * optimizer.param_groups[0]["lr"],
                )

        logger.log(
            {"Test/MultiClass/LearningRate": optimizer.param_groups[0]["lr"]},
            step=epoch,
        )

        metrics = test(network, test_loader, criterion, device)
        report = report_and_log(metrics, class_names, epoch, logger)

        if scheduler is not None:
            scheduler.step()

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = report["accuracy"]
            best_report = report
            best_time = metrics["avg_time"]
            torch.save(
                network.state_dict(),
                os.path.join(args.logdir, "model.pth"),
            )
            utils.save_pickle(network, os.path.join(args.logdir, "model.pkl"))
            n_early_stopping = 0
            logger.log({"Test/MultiClass/BestAccuracy": best_accuracy}, step=epoch)
            logger.log_tensorboard_additional(
                metrics, class_names, step=epoch, tag="best"
            )
        else:
            n_early_stopping += 1

        if n_early_stopping > 100 and early_stopping:
            utils.print_warning(f">> Early stopping at {epoch}")
            break

    if logger.logger_type == "wandb":
        for m in ["Precision", "Recall", "F1-score"]:
            vals = [best_report[x][m.lower()] for x in class_names]
            data = [[name, val] for (name, val) in zip(class_names, vals)]
            table = logger.logger.Table(data=data, columns=["class_names", m])
            logger.log(
                {
                    f"Final {m}": logger.logger.plot.bar(
                        table,
                        "class_names",
                        m,
                        title=f"Per Class {m}",
                    )
                }
            )
    else:
        logger.log_tensorboard_additional(
            metrics, class_names, step=epoch, tag="final_test"
        )

    logger.finish()

    # Summary
    cnn_params = utils.count_parameters(network, no_show=True)
    if is_aln:
        aln_params = network.count_lu_params()
    total_traintime = datetime.timedelta(seconds=time.time() - t0_train)

    utils.print_info("Train Summary:", CR=False, timestamp=False)
    utils.print_info("-" * 100, timestamp=False)
    utils.print_info(f"Model: {args.model}", CR=False, timestamp=False)
    utils.print_info(f"Name: {args.name}", CR=False, timestamp=False)

    if is_aln:
        utils.print_info(
            "NParams (Total, ALN, CNN): "
            f"{aln_params+cnn_params}, {aln_params}, {cnn_params}",
            timestamp=False,
        )
    else:
        utils.print_info(f"NParams: {cnn_params}", timestamp=False)

    utils.print_info(f"Toral train time: {total_traintime}", timestamp=False)
    utils.print_info(f"Best accuracy: {best_accuracy}", CR=False, timestamp=False)
    utils.print_info(
        f"Best model average eval time (per sample, msec): {1000*best_time:0.2}",
        CR=False,
        timestamp=False,
    )
