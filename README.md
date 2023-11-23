# RepAdapter

Official implementation of "[Towards Efficient Visual Adaption via Structural Re-parameterization](https://arxiv.org/pdf/2302.08106.pdf)".
Repadapter is a parameter-efficient and computationally friendly adapter for giant vision models, which can be seamlessly integrated into most
 vision models via structural re-parameterization. Compared to Full Tuning, RepAdapter saves up to 25% training time, 20% GPU memory, and 94.6% storage cost of ViT-B/16 on VTAB-1k.


<p align="center">
	<img src="./misc/RepAdapter.jpg" width="1000">
</p>

## Updates 
- (2023/02/16) Release our RepAdapter project.

## Data Preparation
We provide two ways for preparing VTAB-1k:
- Download the source datasets, please refer to [NOAH](https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation).
- We provide the prepared datasets, which can be download from  [google drive](https://drive.google.com/file/d/1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p/view?usp=share_link).

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

## Usage Example
The following is a simple example of using RepAdapter to load and train a model

1. Import repadapter.py from module
```python 
from repadapter import set_repadapter, save_repadapter,load_repadapter
``` 

2. Insert RepAdapter layers into all linear layers in the model
```python
set_repadapter(model=model)
``` 
If you need to train only specific linear layers, you can modify set_repadapter to use regular expressions to match specific names.
```python
import re
import torch.nn as nn
def set_repadapter(model, pattern):
    # Compile regular expression patterns
    regex = re.compile(pattern)
    for name, module in model.named_modules():
        # Check if the module is a linear layer and if the name matches a regular expression
        if isinstance(module, nn.Linear) and regex.match(name):
```

3. Set the requires_grad attribute of the model parameters.
- Set the requires_grad attribute of the model parameters as needed to determine which parameters require training.
```python
trainable = []
for n, p in model.named_parameters():
    if any([x in n for x in ['repadapter']]):
        trainable.append(p)
        p.requires_grad = True
    else:
        p.requires_grad = False
```

4. Save the checkpoint of RepAdapter
- After training is completed, generally only the repadapter parameters of the model are saved. This can save a significant amount of disk space, which is one of the advantages of using RepAdapter.
```python
import os
save_repadapter(os.path.join(output_dir,"final.pt"), model=model)
```

5. Load the checkpoint of RepAdapter
- If you need to load a model that has been saved after training, the model needs to execute set_repadapter before loading.
```python
load_repadapter(load_path, model=model)
```

6. Reparameterize the model
- merge_repadapter is used after model training to simplify the model structure, reducing the model size and inference time.
- merge_repadapter takes the model and the save path of the repadapter as inputs and performs reparameterization.
```python
merge_repadapter(model,load_path=None,has_loaded=False)
```


## Citation

If this repository is helpful for your research, or you want to refer the provided results in your paper, consider cite:
```BibTeX
@article{luo2023towards,
  title={Towards Efficient Visual Adaption via Structural Re-parameterization},
  author={Luo, Gen and Huang, Minglang and Zhou, Yiyi  and Sun, Xiaoshuai and Jiang, Guangnan and Wang, Zhiyu and Ji, Rongrong},
  journal={arXiv preprint arXiv:2302.08106},
  year={2023}
}
```
 
