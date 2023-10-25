### Intro

This is a PyTorch implementation used for video task. 

### Catalog

- [x] Video code

### Usage

#### Install
* Tesla V100 (32G): CUDA 10.1 + PyTorch 1.6.0 + torchvision 0.7.0
* timm 0.4.8
* einops
* easydict

#### Data Preparation
See [DATASET.md](DATASET.md).

#### Training
Start
```bash
# video
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=8 \
    --node_rank=$1 --master_addr=$2 --master_port=22234 \
    --use_env main_video.py \
    --finetune /path/to/pre_trained/checkpoints \
    --output_dir /path/to/output \
    --batch_size 16 --epochs 90 --blr 0.1 --weight_decay 0.0 --dist_eval \
    --data_path /path/to/SSV2 --data_set SSV2 \
    --ffn_adapt
```
on each of 8 nodes. `--master_addr` is set as the ip of the node 0. and `--node_rank` is 0, 1, ..., 7 for each node.



To obtain the pre-trained checkpoint, see [PRETRAIN.md](PRETRAIN.md).
### Acknowledgement

The project is based on [MAE](https://github.com/facebookresearch/mae), [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [timm](https://github.com/rwightman/pytorch-image-models), and [MAM](https://github.com/jxhe/unify-parameter-efficient-tuning).
Thanks for their awesome works.


### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.
