# ICLR '21 DATA

Working dir: `/shared/nitishg/data/drop/iclr21`

### ICLR 20 DROP data-subset
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
``` 

### QDMR data
`make_qdmr_data.sh` -- outputs `qdmr-v1` directory with qdmr-drop data (w/ execution supervision)

```
/shared/nitishg/data/drop/iclr21/qdmr-v1

Train:
Total num paras: 2756  questions:4762

Dev:
Total num paras: 229  questions:773
```

### Split and Merge
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
```
