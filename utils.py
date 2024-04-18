# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
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
# limitations under the License.

import random
import yaml
import time
import numpy as np
import torch


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def throughput(model, img_size=224, bs=1):
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size, img_size).cuda()
        batch_size = x.shape[0]
        # model=create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
        model.eval()
        for i in range(50):
            model(x)
        torch.cuda.synchronize()
        print(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('memory:', torch.cuda.max_memory_allocated() / MB)


@torch.no_grad()
def save(model_folder, method, dataset, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable[n] = p.data
    torch.save(trainable, './%s/%s/%s.pt' % (model_folder, method, dataset))
    with open('./%s/%s/%s.log' % (model_folder, method, dataset), 'w') as f:
        f.write(str(ep) + ' ' + str(acc))


def load(model_folder, method, dataset, model):
    model = model.cpu()
    st = torch.load('./%s/%s/%s.pt' % (model_folder, method, dataset))
    model.load_state_dict(st, False)
    return model

def get_config(method, dataset_name, few_shot=False):
    if few_shot:
        config_name = './configs/%s_few_shot/%s.yaml' % (method, dataset_name)
    else:
        config_name = './configs/%s/%s.yaml' % (method, dataset_name)
    with open(config_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config