# Neural Module Networks for Reasoning over Text

This is the official code for the ICLR 2020 paper, [Neural Module Networks for Reasoning over Text](https://arxiv.org/abs/1912.04971).
This repository contains the code for replicating our experiments and can be used to extend our model as you wish.

## Resources
1. Download the data and a trained model checkpoint from [here](https://drive.google.com/drive/folders/1ZPnQqQHBrWXEF4z3yTK5wL5sCI8gG98T?usp=sharing).
Contains a directory called `iclr_cameraready` (1.2GB).
 Place the directory on any accessible path, henceforth referred to as -- `MODEL_CKPT_PATH`
 
2. Clone the `allennlp-semparse` repository from [here](https://github.com/allenai/allennlp-semparse).
Path to this would be referred to as -- `PATH_TO_allennlp-semparse`


## Installation
The code is written in python using [AllenNLP](https://github.com/allenai/allennlp) and
[allennlp-semparse](https://github.com/allenai/allennlp-semparse).

```
conda create -name nmn-drop python=3.6 
conda activate nmn-drop
pip install allennl==0.9   # create conda environment 
git clone git@github.com:nitishgupta/nmn-drop.git
cd nmn-drop/
mkdir resources; cd resources; ln -s MODEL_CKPT_PATH/iclr_cameraready .; cd ..    
ln -s PATH_TO_allennlp-semparse/allennlp-semparse/allennlp_semparse/ ./ 
```


