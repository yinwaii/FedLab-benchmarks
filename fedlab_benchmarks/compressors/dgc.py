# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
import random
from .base_compressor import Memory, Compressor


class DGCSGDMemory(Memory):
    """ Memory for momentum correction in DGC for momentum SGD optimizer"""
    def __init__(self,
                 momentum=0.9,
                 nesterov=False,
                 gradient_clipping=None,
                 momentum_masking=True):
        self.gradient_clipping = gradient_clipping
        self.momentum_masking = momentum_masking

        self.momentum = momentum
        self.nesterov = nesterov
        self.momentums = {}
        self.velocities = {}

    def initialize(self, named_parameters):
        for name, param in named_parameters:
            self.momentums[name] = torch.zeros_like(param.data)
            self.velocities[name] = torch.zeros_like(param.data)

    def compensate(self, grad, name, accumulate=True):
        """Update the velocities with the momentums."""
        if self.gradient_clipping is not None:
            grad = self.gradient_clipping(grad)
        mmt = self.momentums[name]
        if accumulate:
            vec = self.velocities[name]
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                vec.add_(mmt).add_(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                vec.add_(mmt)
            return vec
        else:
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                return mmt.add(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                return mmt.clone()  # TODO: save this clone

    def update(self, name, ctx):
        """Update the momentums."""
        indices = ctx[0]
        if self.momentum_masking:
            self.momentums[name].view(-1).index_fill_(
                0, indices, 0
            )  # index_fill_(dim,index,val)按照参数index总的索引数确定的顺序，将原tensor用参数val值填充
        self.velocities[name].view(-1).index_fill_(0, indices, 0)

    def state_dict(self):
        return dict(momentums=self.momentums, velocities=self.velocities)

    def load_state_dict(self, state_dict):
        momentums = state_dict['momentums']
        velocities = state_dict['velocities']
        for name in self.momentums.keys():
            if name in momentums:
                self.momentums[name] = momentums[name]
                self.velocities[name] = velocities[name]


class DGCCompressor(Compressor):
    def __init__(self,
                 compress_ratio,
                 memory=None,
                 sample_ratio=0.01,
                 strided_sample=True,
                 compress_upper_bound=1.3,
                 compress_lower_bound=0.8,
                 max_adaptation_iters=10,
                 resample=True,
                 fp16_values=False,
                 int32_indices=False,
                 warmup_epochs=-1,
                 warmup_coeff=None):

        self.fp16_values = fp16_values
        self.int32_indices = int32_indices

        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.memory = Memory if memory is None else memory
        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            if warmup_coeff is None:
                self.warmup_coeff = self.base_compress_ratio \
                    ** (1. / (self.warmup_epochs + 1))
            else:
                if isinstance(warmup_coeff, (tuple, list)):
                    assert len(warmup_coeff) >= self.warmup_epochs
                    for wc in warmup_coeff:
                        assert 0 < wc <= 1
                else:
                    assert 0 < warmup_coeff <= 1
                self.warmup_coeff = warmup_coeff
        else:
            self.warmup_coeff = 1

        self.sample_ratio = min(max(sample_ratio, 0.01), 1.0)
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.resample = resample

        self.attributes = {}

    def initialize(self, named_parameters):

        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            if self.sample_ratio < 1.0:
                pct_numel = int(math.ceil(numel * self.sample_ratio))
                cpr_numel = int(math.ceil(2 / self.compress_ratio))
                if numel <= cpr_numel:
                    sample_stride = 1
                    num_samples = numel
                else:
                    sample_stride = int(
                        math.ceil(
                            numel / max(pct_numel, cpr_numel) / 32)) * 32 + 1
                    num_samples = numel // sample_stride
                    while num_samples < max(pct_numel, cpr_numel):
                        sample_stride = sample_stride - 8
                        num_samples = numel // sample_stride
            else:
                sample_stride = 1
                num_samples = numel
            top_k_samples = int(math.ceil(num_samples * self.compress_ratio))
            num_selects = int(math.ceil(numel * self.compress_ratio))
            self.attributes[name] = (numel, shape, num_selects, num_samples,
                                     top_k_samples, sample_stride)

    def warmup_compress_ratio(self, epoch):
        if self.warmup_epochs > 0:
            if epoch < self.warmup_epochs:
                if isinstance(self.warmup_coeff, (tuple, list)):
                    compress_ratio = self.warmup_coeff[epoch]
                else:
                    compress_ratio = max(self.warmup_coeff**(epoch + 1),
                                         self.base_compress_ratio)
            else:
                compress_ratio = self.base_compress_ratio
        else:
            compress_ratio = self.base_compress_ratio
        if compress_ratio != self.compress_ratio:
            self.compress_ratio = compress_ratio
            self.initialize(self.attributes.items())

    def _sparsify(self, tensor, name):
        tensor = tensor.view(-1)
        numel, shape, num_selects, num_samples, top_k_samples, sample_stride = self.attributes[
            name]

        importance = tensor.abs()
        if numel == num_samples:
            samples = importance
        else:
            if self.strided_sample:
                sample_start = random.randint(0, sample_stride - 1)
                samples = importance[sample_start::sample_stride]
            else:
                samples = importance[torch.randint(0,
                                                   numel, (num_samples, ),
                                                   device=tensor.device)]

        threshold = torch.min(
            torch.topk(samples, top_k_samples, 0, largest=True,
                       sorted=False)[0])
        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)
        num_indices = indices.numel()

        if numel > num_samples:
            for _ in range(self.max_adaptation_iters):
                if num_indices > num_selects:
                    if num_indices > num_selects * self.compress_upper_bound:
                        if self.resample:
                            indices = indices[torch.topk(importance[indices],
                                                         num_selects,
                                                         0,
                                                         largest=True,
                                                         sorted=False)[1]]
                            break
                        else:
                            threshold = threshold * self.compress_upper_bound
                    else:
                        break
                elif num_indices < self.compress_lower_bound * num_selects:
                    threshold = threshold * self.compress_lower_bound
                else:
                    break
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero().view(-1)
                num_indices = indices.numel()

        indices = indices[:num_selects]
        values = tensor[indices]
        return values, indices, numel, shape, num_selects

    def compress(self, tensor, name):
        # 对于已注册的数据结构/模型参数压缩
        if self.compress_ratio < 1.0 and name in self.attributes:
            # compress
            tensor_compensated = self.memory.compensate(tensor,
                                                        name,
                                                        accumulate=True)
            values, indices, numel, shape, num_selects = \
                self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices, ))
            indices = indices.view(-1, 1)
            values = values.view(-1, 1)

            ctx = (name, numel, shape, values.dtype, indices.dtype,
                   tensor.data.view(numel))
            if self.fp16_values and values.dtype.is_floating_point:
                values = values.type(torch.float16)
            if self.int32_indices and not indices.dtype.is_floating_point:
                indices = indices.type(torch.int32)
            return (values, indices), ctx
        # 对未注册不进行操作 返回置空数据
        else:
            ctx = (name, None, None, tensor.dtype, None, None)
            if self.fp16_values and tensor.dtype.is_floating_point:
                tensor = tensor.type(torch.float16)
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if self.compress_ratio < 1.0 and name in self.attributes:
            # decompress
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.view(-1)
            indices = indices.view(-1)
            if self.fp16_values and vdtype.is_floating_point:
                values = values.type(vdtype)
            if self.int32_indices and not idtype.is_floating_point:
                indices = indices.type(idtype)
            grad.zero_().index_put_([indices], values, accumulate=True)

            return grad.view(shape)
        else:
            if self.fp16_values and vdtype.is_floating_point:
                tensor = tensor.type(vdtype)
            return self.memory.compensate(tensor, name, accumulate=False)