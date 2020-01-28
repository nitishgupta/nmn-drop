# Neural Module Networks for Reasoning over Text

This is the official code for the ICLR 2020 paper, [Neural Module Networks for Reasoning over Text](https://arxiv.org/abs/1912.04971).
This repository contains the code for replicating our experiments and can be used to extend our model as you wish.

## Resources
1. Download the data and a trained model checkpoint from [here](https://drive.google.com/drive/folders/1ZPnQqQHBrWXEF4z3yTK5wL5sCI8gG98T?usp=sharing).
Unzip the downloaded contents and place the resulting directory `iclr_cameraready` inside a convenient location, 
henceforth referred to as -- `MODEL_CKPT_PATH`
 
2. Clone the `allennlp-semparse` repository from [here](https://github.com/allenai/allennlp-semparse) to a convenient location,
 henceforth referred to as -- `PATH_TO_allennlp-semparse`.
 Checkout using `git checkout 937d594` the specific commit that this code is built on. 
 *Such issues will be resolved soon when allennlp-semparse becomes pip installable*.


## Installation
The code is written in python using [AllenNLP](https://github.com/allenai/allennlp) and
[allennlp-semparse](https://github.com/allenai/allennlp-semparse).

The following commands create a miniconda environment, install allennlp, and creates symlinks for allennlp-semparse and the downloaded resources.
```
# Make conda environment
conda create -name nmn-drop python=3.6
conda activate nmn-drop

# Install required packages
pip install allennlp==0.9
pip install dateparser==0.7.2
python -m spacy download en_core_web_lg

# Clone code and make symlinks
git clone git@github.com:nitishgupta/nmn-drop.git
cd nmn-drop/
mkdir resources; cd resources; ln -s MODEL_CKPT_PATH/iclr_cameraready ./; cd ..    
ln -s PATH_TO_allennlp-semparse/allennlp-semparse/allennlp_semparse/ ./ 
```

## Prediction
To make predictions on your data, format your data in a [json lines format](http://jsonlines.org/) -- `input.jsonl`
where each line is a valid JSON value containing the keys `"question"` and `"passage"`.

Run the command
```
allennlp predict \
    --output-file test/output.jsonl \
    --predictor drop_demo_predictor \
    --include-package semqa \
    --silent \
    --batch-size 1 \ 
    resources/iclr_cameraready/ckpt/model.tar.gz \
    test/input.jsonl
```
The output `output.jsonl` contains the answer in an additional key `"answer"`.  
 
## Evaluation
To evaluate the model on the dev set, run the command -- `bash scripts/iclr/evaluate.sh` 

The `model_ckpt`/`data` path in the script can be modified to evaluate a different model on a different dataset.

## Visualization
To generate text based visualization of the model's prediction on the development data, run the command -- `bash scripts/iclr/predict.sh`

A file `drop_mydev_verbosepred.txt` is written to `MODEL_CKPT_PATH/iclr_cameraready/ckpt/predictions` containing this visualization.

*An interactive demo of our model will be available soon.*

## Training
We already provide a trained model checkpoint and the subset of the DROP data used in the ICLR2020 paper with the resources above.

If you would like to re-train the model on this data, run the command -- `bash scripts/iclr/train.sh`.

The model checkpoint would be saved at `MODEL_CKPT_PATH/iclr_cameraready/my_ckpt`.

Note that this code needs the DROP data to be preprocessed with additional information such as, tokenization, numbers, and dates, etc.
To train a model on a different subset of the DROP data, 
this pre-processing can be performed using the python script `datasets/drop/preprocess/tokenize.py` on any DROP-formatted `json` file.

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

If you'd like to contribute code, feel free to open a [pull request](https://github.com/nitishgupta/nmn-drop/pulls).
If you find an issue with the code or require additional support, please open an [issue](https://github.com/nitishgupta/nmn-drop/issues).
