# ICLR '21 DATA

Working dir: `/shared/nitishg/data/drop/iclr21`

## ICLR 20 DROP data-subset
`make_iclr20_data.sh` -- This should make directory `iclr20_full` with drop-subset used in iclr20 paper 
(w/ execution supervision for date-compare and HMYW).

```
/shared/nitishg/data/drop/iclr21/iclr20_subsets/iclr20_full-v1

Train
Number of input passages: 3881
Number of input questions: 19204
Dev
Number of input passages: 396
Number of input questions: 2141

V2 - without data augmentation for date-compare
Train
Number of input passages: 3881
Number of input questions: 16805
Dev
Number of input passages: 396
Number of input questions: 1841
``` 

#### ICLR20 w/ Filter
In the `make_iclr20_data.sh` comment the `postprocessing`


## QDMR data
`make_qdmr_data.sh` -- outputs `qdmr-v1` directory with qdmr-drop data (w/ execution supervision)

```
/shared/nitishg/data/drop/iclr21/qdmr-v1

Train:
Total num paras: 2756  questions:4762

Dev:
Total num paras: 229  questions:773

v4 - w/ Filter
Train:
Total num paras: 2783  questions:4819
Dev:
Total num paras: 232  questions:782
```

#### QDMR w/ Filter
Don't use Filter module from QDMR annotations, it's quite noisy. Use the `--remove_filter_module` arg in 
`process_drop_qdmr.py`.

After that step, use `datasets.drop.iclr21.add_filter` to add filter. Only adds in about 120ish examples though.

## Split and Merge -- iclr_qdmr-v#
`split_and_merge.sh` -- 
1. Splits iclr20 and qdmr train data into train/dev and convert dev into test.
2. Merge these into `iclr_qdmr-v1` dataset
3. Create version `iclr_qdmr-v1-noexc` by removing intermedate execution supervision

```
ICLR20 split
Before split: P: 3881 Q: 19204
After Split:
Train P: 3493 Q: 17228
Dev: P:388 Q:1976
Test data; P: 396  Q:2141

QDMR split
Before split: P: 2756 Q: 4762
After Split:
Train P: 2343 Q: 4062
Dev: P:413 Q:700
Test data; P: 229  Q:773

iclr_qdmr-v1 merged data:
Train: passages: 4237 questions: 20443   {'program_supervision': 20443, 'execution_supervised': 4146}
Dev:   passages: 771  questions: 2666    {'program_supervision': 2666, 'execution_supervised': 531}
Test:  passages: 446  questions: 2747    {'program_supervision': 2747, 'execution_supervised': 505}

iclr_qdmr-v2 merged data:
Train: passages: 4237 questions: 18283  {'program_supervision': 18283, 'execution_supervised': 2546}
Dev:   passages: 771  questions: 2427   {'program_supervision': 2427, 'execution_supervised': 360}
Test:  passages: 446  questions: 2447   {'program_supervision': 2447, 'execution_supervised': 274}

iclr_qdmr-v4 w/ filter merged data:
Train: passages: 4241 questions: 18299  {'program_supervision': 18299, 'execution_supervised': 2554}
Dev:   passages: 779  questions: 2460   {'program_supervision': 2460, 'execution_supervised': 348}
Test:  passages: 447  questions: 2456   {'program_supervision': 2456, 'execution_supervised': 274}
```

## Compositional split

#### Complex diff split
```
python -m datasets.compositional_split.complexdiff_qsplit \
    --input_dir /shared/nitishg/data/drop/iclr21/iclr_qdmr-v3-noexc \
    --output_dir /shared/nitishg/data/drop/iclr21/diff_compsplit-v3
```

#### Filter split
```
python -m datasets.compositional_split.filter_qsplit \
    --input_dir /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc \
    --output_dir /shared/nitishg/data/drop/iclr21/filter_compsplit-v4
```


## Paired Data - Constructed
Main script -- `construct_paired_examples.py`
Helper script contains some useful functions -- `datasets.drop.paired_data.generate_diff_questions`

The question-types for which paired data needs to be generated is set manually in a dictionary in script.
Need to figure out a better way for this

```
python -m datasets.drop.paired_data.construct_paired_examples \
    --input_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train.json \
    --output_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-CONS.json
```

## Paired Data - Found within the dataset
Find find-nodes within a single passage's questions that match enough (using bert-score and ner-match).
Add best matching select-node, above a threshold, as paired example. 

```
python -m datasets.drop.paired_data.discover_paired_examples \
    --input_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train.json \
    --strpair_f1_tsv /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/strpair2f1.tsv \
    --output_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-FOUND-06.json
```

## Paired Data - Model-generated
First generate augmented num-date questions for non-football passages

This requires `model.tar.gz` for a pre-trained BART-based question-generation model.
Currently this model path is hardcoded.
```
python -m datasets.drop.data_augmentation.generate_numdate_questions \
    --input_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train.json \
    --output_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/numdate_aug.json
```

Find paired examples for DROP questions from augmented questions
```
python -m datasets.drop.paired_data.modelgen_paired_examples \
    --drop_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train.json 
    --aug_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/numdate_aug.json \
    --strpair_f1_tsv /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/numdate-aug-strpair2f1.tsv \
    --output_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-MODGEN-06.json
```
 

### Paired data - merge
Merging two kinds of paired data `drop_dataset_train.json`
```
python -m datasets.drop.paired_data.merge_datasets \
    --input_json1 /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-FGS-DCYD-ND-MM-v2.json \
    --input_json2 /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-FOUND-06.json \
    --output_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-CONS-FOUND-06.json

python -m datasets.drop.paired_data.merge_datasets \
    --input_json1 /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-FGS-DCYD-ND-MM-v2.json \
    --input_json2 /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-MODGEN-06.json \
    --output_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-CONS-MODGEN-06.json

python -m datasets.drop.paired_data.merge_datasets \
    --input_json1 /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-FOUND-06.json \
    --input_json2 /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-MODGEN-06.json \
    --output_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-FOUND-MODGEN-06.json

python -m datasets.drop.paired_data.merge_datasets \
    --input_json1 /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-CONS-MODGEN-06.json \
    --input_json2 /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-FOUND-06.json \
    --output_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_train-CONS-MODGEN-FOUND-06.json
```


## Faithful Annotations
Two scripts, to handle filter or not. 

The DROP subset being used should contain questions annotated in ACL (drop-dev-set).

Since in ICLR21 splits, we've made the drop-dev as test set, we're using that for `drop_input_json`. 
Be careful of the version being used, w/ filter or w/o filter.
```
python -m datasets.drop.faithful.convert_acl_annotations_filter \
    --acl_json /shared/nitishg/data/drop/iclr21/faithful/acl_faithful_annotations.json \
    --drop_input_json /shared/nitishg/data/drop/iclr21/iclr_qdmr-v4-noexc/drop_dataset_test.json \
    --output_json /shared/nitishg/data/drop/iclr21/faithful/iclr21_filter_faithful.json
```
