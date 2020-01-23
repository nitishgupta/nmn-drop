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
pip install allennlp==0.9 
git clone git@github.com:nitishgupta/nmn-drop.git
cd nmn-drop/
mkdir resources; cd resources; ln -s MODEL_CKPT_PATH/iclr_cameraready .; cd ..    
ln -s PATH_TO_allennlp-semparse/allennlp-semparse/allennlp_semparse/ ./ 
```

## Evaluation
Run the command `bash scripts/iclr/evaluate.sh` to evaluate the model on the dev set.

The model_ckpt/data path in the script can be modified to evaluate a different model on a different dataset.

## Visualization
The command `bash scripts/iclr/predict.sh` can be used to generate text based visualization of the model's prediction on the development data.
A file `drop_mydev_verbosepred.txt` is written to `MODEL_CKPT_PATH/iclr_cameraready/ckpt/predictions` containing this.

An interactive demo of our model will be available soon.

## References
Please consider citing our work if you found this code or our paper beneficial to your research.

```
@inproceedings{nmn:iclr20,
  author = {Nitish Gupta and Kevin Lin and Dan Roth and Sameer Singh and Matt Gardner},
  title = {Neural Module Networks for Reasoning over Text},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2020}
}
```

## Contributions and Contact
This code was developed by [Nitish Gupta](https://nitishgupta.github.io), contact [nitishg@seas.upenn.edu](mailto:nitishg@seas.upenn.edu).

Feel free to open a [pull request](https://github.com/nitishgupta/nmn-drop/pulls) if you'd like to contribute code.
If you find an issue with the code or require additional support, please open an [issue](https://github.com/nitishgupta/nmn-drop/issues).
