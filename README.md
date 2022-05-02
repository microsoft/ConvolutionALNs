# Convolutional Adaptive Logic Networks (CALNs)

## Introduction

This repository implements a minimum necessary codebase for creating, training and evaluating CALNs documented
in the paper: [Convolutional Adaptive Logic Networks: A First Approach](https://github.com/microsoft/ConvolutionALNs/blob/main/aln_paper.pdf).

## Instructions for usage

### Requirements

The code in this repository is tested under the following requirements:
 - pytorch: 1.11.0 (stable)
 - cuda: 10.2
 - tensorboard: 2.0.0
 - scikit-learn: 1.0.2
 - matplotlib: 3.5.1
 - prettytable: 3.2.0
 - wandb: 0.12.15 (only required if WandB is used for logging and monitoring)

### Installation

This is a development codebase. Therefore, it is suggested to install the code in the caln subdirectory as a Python development package:

```

cd caln
pip install -e .

```

### General usage and guidelines

The main class that implements CALN is `ConvolutionALNet` in core/caln.py. A sample usage is given in trainings/models.py. Basically, `ConvolutionALNet` receives a backbone and attaches an ALN at its end. Note that the ALN weights are not updated by a gradient descent method. However, the gradients are propagated through the ALN back to the backbone weights.

`forward` method receives a tensor and runs the entire network to produce the output.
`adapt` should be used at each training iteration to update ALN weights.
`grow` should be used at split iteration.

The main code, that trains a variety of CALNs on CIFAR-10 dataset is in caln/trainings/train_cifar10.py. The command line input options to this script is described in training/common_utils.py. You can also see the list by entering `$ python train_cifar10.py` at the command line in trainings/ folder. Note that train_cifar10.py can also be used to train a couple of ResNet architectures. You can use the following command lines to approximately reproduce the reported results in the paper (all of the experiments run on GPU):

* ResNet13+ALN:
`python train_cifar10.py --name CALN_ResNet13_Cifar10 --model CALN --optimizer SGD --epochs 1000 --lr 0.1 --aln_lr 0.01 --init_pieces 3 --root_op min --split_step 15 --max_splits 1  --split_step_increment 2 --device cuda:0`

* ResNet14:
`python train_cifar10.py --name ResNet14_Cifar10 --model ResNet14 --optimizer SGD --epochs 1000 --device cuda:0`

* ResNet18:
`python train_cifar10.py --name ResNet18_Cifar10 --model ResNet18 --optimizer SGD --epochs 1000 --device cuda:0`

The training logs are stored in the default (running) folder at ./[NAME]/, e.g. for ResNet13+ALN, it is stored in ./CALN_ResNet13_Cifar10. Use `--logdir` to change the default `./`. To prevent losing past experiments, it is not allowed to overwrite on an existing folder. The only folder that can be overwrriten automatically is `test`. That is, one can use `test` as the experiment name and the previous test experiments are overwritten.

The CIFAR-10 dataset is sought in the root folder `./`, if it does not exist, it automatically downloads it. To change the path for CIFAR-10 dataset use `--cifar10_path [NEW_PATH]` argument.

To monitor the training process, the default tool is [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html). However, WandB is also supported and we strongly suggest to use [WandB](https://wandb.ai/site) because hyperparameters and more details are logged using WandB. An example usage with WandB is:

`python train_cifar10.py --name CALN_ResNet13_Cifar10 --model CALN --optimizer SGD --epochs 1000 --lr 0.1 --aln_lr 0.01 --init_pieces 3 --root_op min --split_step 15 --max_splits 1  --split_step_increment 2 --device cuda:0 --logger wandb --wandb_project [PROJECT_NAME] --wandb_entity [ENTITY_NAME]`,

where [PROJECT_NAME] and [ENTITY_NAME] must be set properly (refer to [WandB documentation](https://docs.wandb.ai/).

### Known issues and suggestions for improvements

The main bottleneck of the current implementation is the for-loop usage while evaluating and adapting the ALNs. Because the ALNs are independent, in principle it is straightforward to run them in parallel. This will significantly reduces train and evaluation time and it also brings the possibility to run the CALNs on datasets with larger number of classes.

We have not tested any form of training and evaluation across multiple GPUs (model or data parallel).



## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

