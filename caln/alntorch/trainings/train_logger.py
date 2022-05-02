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
from torch.utils.tensorboard import SummaryWriter

from alntorch.core.utils import (
    plot_confusion_matrix,
)


class TrainLogger:
    def __init__(self, logger_type, logdir, **kwargs):
        logger_type = logger_type.lower()

        if logger_type not in ["tensorboard", "wandb"]:
            raise ValueError("Only available support for Tensorboard and WandB.")

        self._type = logger_type
        if self._type == "wandb":
            project = kwargs.get("wandb_project", None)
            entity = kwargs.get("wandb_entity", None)
            if project == "" or entity == "":
                raise ValueError(
                    "To use wandb, an entity name and a project name must be provided."
                )

            # Only import wandb if the user want to use it for logging.
            import wandb

            wandb.login()
            wandb.init(
                project=project,
                entity=entity,
                config=kwargs,
            )
            self.logger = wandb

        elif self._type == "tensorboard":
            self.logger = SummaryWriter(log_dir=logdir)

    @property
    def logger_type(self):
        return self._type

    def log(self, args_dict, step=0):
        if self._type == "wandb":
            self.logger.log(args_dict)
        elif self._type == "tensorboard":
            for k1, v1 in args_dict.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        self.logger.add_scalar(f"{k1}{k2}", v2, step)
                else:
                    self.logger.add_scalar(k1, v1, step)

    def log_tensorboard_additional(self, metrics, class_names, step, tag):
        if self._type == "tensorboard":
            self._log_confusion_matrix(metrics, class_names, step, tag)
            self._log_pr_curves(metrics, class_names, step, tag)

    def finish(self):
        if self._type == "wandb":
            self.logger.finish()

    def _log_confusion_matrix(self, metrics, class_names, step=None, tag="val"):
        fig = plot_confusion_matrix(
            metrics["y_true"],
            metrics["y_pred"],
            labels=class_names,
        )
        self.logger.add_figure(f"{tag}/confusion_matrix", fig, step)

    def _log_pr_curves(self, metrics, class_names, step=None, tag="val"):
        for class_id, class_name in enumerate(class_names):
            class_score = np.array(metrics["y_pred"]) == class_id
            class_label = np.array(metrics["y_true"]) == class_id

            self.logger.add_pr_curve(
                f"{tag}/{class_name}",
                labels=class_label,
                predictions=class_score,
                global_step=step,
            )
