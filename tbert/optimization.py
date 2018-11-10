# The MIT License
# Copyright 2019 Innodata Labs and Mike Kroutikov
#
# PyTorch port of
# https://github.com/google-research/bert/modeling.py
#
# Original code copyright follows:
#
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import json
#
from torch.optim.lr_scheduler import LambdaLR


class LinearDecayWithWarpupLR(LambdaLR):

    def __init__(self, optimizer, train_steps, warmup_steps, last_epoch=-1):

        def schedule(step):
            if step <= warmup_steps:
                return step / warmup_steps
            assert step <= train_steps
            return (train_steps - step) / (train_steps - warmup_steps)

        LambdaLR.__init__(self, optimizer, schedule, last_epoch=last_epoch)
