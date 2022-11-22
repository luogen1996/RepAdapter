# RepAdapter

Official implementation of "Towards Efficient Visual Adaption via Structural Re-parameterization".
Repadapter is a parameter-efficient and computationally friendly adapter for giant vision models, which can be seamlessly integrated into most
 vision models via structural re-parameterization.


<p align="center">
	<img src="./misc/RepAdapter.jpg" width="550">
</p>

## Updates 
- (2022/11/22) Release our RepAdapter project.

## Data Preparation
We provide two ways for preparing the datasets:
- Download the source datasets, please refer to [NOAH](https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation).
- We provide the prepared datasets, which can be download from [here]().

After that, the file structure should look like:
```
$ROOT/data
|-- cifar
|-- caltech101
......
|-- diabetic_retinopathy
```
 
- Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `./ViT-B_16.npz`

## Training and Evaluation
1. Search the hyper-parameter s for RepAdapter (optional)
```sh 
bash search_repblock.sh
``` 

2. Train RepAdapter
```sh 
bash train_repblock.sh
``` 

3. Test RepAdapter
```sh 
python test.py --method repblock --dataset <dataset-name> 
```

### Performance on VTAB-1k (seed = 42)
####Overall performance on vtab-1k

| Methods         | Natural | Specialized | Structured | Average |     logs     |  checkpoints |
|-----------------|:-------:|:-----------:|:----------:|:-------:|:------------:|:------------:|
| RepAdapter_attn |   82.63 |       85.87 |      62.86 |   77.12 | (onedrive)[] | (onedrive)[] |
| RepAdapter      |  82.87  |       86.04 |      63.29 |  77.40  | (onedrive)[] | (onedrive)[] |

####Performance of each dataset

| Methods         | cifar | caltech101 |  dtd  | flower102 |  pets |  SVHN | sun397 | Camelyon | EuroSAT | Resisc45 | Retinopathy | Clevr-count | Clevr-dist | DMLab | Kitti-dist | Dspr-loc | Dspr-ori | sNORB-Azim | sNORB-Ele |
|-----------------|:-----:|:----------:|:-----:|:---------:|:-----:|:-----:|:------:|:--------:|:-------:|:--------:|:-----------:|:-----------:|:----------:|:-----:|:----------:|:--------:|:--------:|:----------:|:---------:|
| RepAdapter_attn | 75.7  |    92.2    | 72.5  |   99.3    | 92.2  | 90.8  |  57.5  |   86.9   |  96.0   |   85.7   |    75.5     |    82.4     |    64.1    | 52.3  |    81.9    |   86.6   |   54.3   |    38.0    |   46.9    |
| RepAdapter      | 75.4  |    91.2    | 73.9  |   99.3    | 91.5  | 89.8  |  57.6  |   87.0   |  95.9   |   85.3   |    75.4     |    83.0     |    64.8    | 53.1  |    81.6    |   85.6   |   53.5   |    35.4    |   46.0    |
 
