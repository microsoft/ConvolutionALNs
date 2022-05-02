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

import torch


def check_device(device="cpu", return_info=False):
    device = device.lower()

    if return_info:
        device_info = {"total_gpu": torch.cuda.device_count()}
        for i in torch.cuda.device_count():
            device_info[i] = torch.cuda.get_device_name(i)

    if device.startswith("cuda:"):
        try:
            device_no = int(device.split(":")[-1])
        except BaseException:
            raise ValueError("Invalid device number.")

        if not torch.cuda.is_available():
            raise ValueError("No Cuda device found.")
        else:
            if device_no > torch.cuda.device_count() - 1:
                raise ValueError("GPU ID exeeds number of GPUs.")
    elif device != "cpu":
        raise ValueError(f"Invalid device name {device}.")

    if return_info:
        return device, device_info
    else:
        return device
