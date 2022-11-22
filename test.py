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


from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from argparse import ArgumentParser
from dataset import *
from utils import *
from repadapter import set_RepAdapter, set_RepWeight


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    pbar = tqdm(dl)
    model = model.cuda()
    for batch in pbar:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 0)

    return acc.result()[0]




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k',
                        choices=['vit_base_patch16_224_in21k', 'swin_base_patch4_window7_224_in22k',
                                 'convnext_base_22k_224'])
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='repblock', choices=['repattn,repblock'])
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--dim', type=int, default=8)
    args = parser.parse_args()
    print(args)
    config = get_config(args.method, args.dataset)

    # build model
    if 'vit' in args.model:
        model = create_model(args.model, drop_path_rate=0.1,pretrained=True)
    elif 'swin' in args.model:
        model = create_model(args.model, drop_path_rate=0.1,pretrained=True)
    elif 'conv' in args.model:
        model = create_model(args.model, drop_path_rate=0.1,pretrained=True)
    else:
        assert NotImplementedError

    # build dataset
    train_dl, test_dl = get_data(args.dataset)

    # running throughput
    model.cuda()
    print('before reparameterizing: ')
    throughput(model)

    # build repadapter
    set_RepAdapter(model, args.method, dim=args.dim, s=args.scale, args=args, set_forward=False)

    # load model
    model.reset_classifier(config['class_num'])
    model = load(args.method, config['name'], model)

    # fusing repadapter
    set_RepWeight(model, args.method, dim=args.dim, s=args.scale, args=args)

    # running throughput
    model.cuda()
    print()
    print('after reparameterizing: ')
    throughput(model)

    # testing loop
    acc = test(model, test_dl)
    print('Accuracy:', acc)

