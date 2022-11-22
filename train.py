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

from torch.optim import AdamW
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from dataset import *
from utils import *
from repadapter import set_RepAdapter
from torch import nn
from timm.data import Mixup
from timm.loss import  SoftTargetCrossEntropy

def train(config, model, dl, opt, scheduler, epoch,mixup_fn=None,criterion=nn.CrossEntropyLoss()):
    model.train()
    model = model.cuda()
    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            if mixup_fn is not None:
                x,y=mixup_fn(x,y)
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(model, test_dl)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                save(config['method'], config['name'], model, acc, ep)
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    #pbar = tqdm(dl)
    model = model.cuda()
    for batch in dl:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 0)

    return acc.result()[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k', choices=['vit_base_patch16_224_in21k','swin_base_patch4_window7_224_in22k','convnext_base_22k_224']) #swin_tiny_patch4_window7_224
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='repblock',choices=['repattn','repblock'])
    parser.add_argument('--scale', type=float, default=0)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--few-shot',  action='store_true')
    parser.add_argument('--shots',   type=int, default=1)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    config = get_config(args.method, args.dataset,args.few_shot)

    #mkdir for logs and models
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./models/%s'%(args.method)):
        os.makedirs('./models/%s'%(args.method))


    if 'vit' in args.model:
        model = create_model(args.model, drop_path_rate=0.1,checkpoint_path='./ViT-B_16.npz')
    elif 'swin' in args.model:
        model = create_model(args.model, drop_path_rate=0.1,pretrained=True)
    elif 'conv' in args.model:
        model = create_model(args.model, drop_path_rate=0.1,pretrained=True)
    else:
        assert NotImplementedError

    model.cuda()
    throughput(model)
    train_dl, test_dl = get_data(args.dataset,few_shot=args.few_shot,mean=model.default_cfg['mean'],std=model.default_cfg['std'])

    set_RepAdapter(model, args.method, dim=args.dim, s=config['scale'] if args.scale==0 else args.scale, args=args)
    model.cuda()
    throughput(model)

    if hasattr(model,'blocks'):
        print(model.blocks[0])
    elif hasattr(model,'layers'):
        print(model.layers[0])
    elif hasattr(model,'stages'):
        print(model.stages[0])
    else:
        assert NotImplementedError

    trainable = []
    model.reset_classifier(config['class_num'])
    
    config['best_acc'] = 0
    config['method'] = args.method
    total=0
    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable.append(p)
            total+=p.nelement()
        else:
            p.requires_grad = False
    print('  + Number of trainable params: %.2fK' % (total / 1e3))  # 每一百万为一个单位
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=100,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)
    if args.few_shot:
        mixup_fn=Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=config['class_num'])
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn=None
        criterion = torch.nn.CrossEntropyLoss()
    model = train(config, model, train_dl, opt, scheduler, epoch=100,mixup_fn=mixup_fn,criterion=criterion)
    print(config['best_acc'])
