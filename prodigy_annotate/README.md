# Prodigy annotation tool for DROP-wQDMR programs

This is done locally on my laptop (can be accomplished using ssh-port forwarding)

### Program diagnostics
Create subsets of the data for different kinds of programs we want to manually analyze for issues. 
This creates different directories for different program-types under `diagnostics`.
For example, `longshort_no_minmax`, etc.

Each directory would contain a `drop_dataset_train.json` and `train.jsonl` where the later contains dicts with the
key `prodigy_lisp` required for annotations using prodigy.
```
python -m analysis.qdmr.program_diagnostics \ 
    --input_dir /shared/nitishg/data/drop-w-qdmr/qdmr-filter \
    --output_root /shared/nitishg/data/drop-w-qdmr/qdmr-filter/diagnostics
```

### Annotating data
Run this from the root directory (nmn-drop-qdmr). We also keep a config file in the root (`prodigy.json`).

```
prodigy drop-recipe filternum_potential \ 
    ~/drop/data/qdmr-filter/diagnostics/filternum_potential/train.jsonl -F prodigy_annotate/recipe.py 
```

Here, `filternum_potential` is the name of the prodigy-dataset that get's created and would contain annotations.

If starting a new annotation process, make sure that this is a new dataset. Otherwise, continue on the same dataset. 

### Retrieving annotations
Output the database (`filternum_potential` above) into a jsonlines format.
```
prodigy db-out  filternum_potential> ~/drop/data/qdmr-filter/diagnostics/filternum_potential/annotations.jsonl
```

This JsonL's dictionaries would contain a key `program` -- containing manual annotated program (default: `prodigy_lisp`)
and `remarks` -- containing manual remarks (`optional`).


### Annotations to TSV
This changes the `program` key to `program_annotation` and also writes out an `annotations.tsv` for viewing in Google 
Sheets.
```
python -m prodigy_annotate.prodigy_to_jsonl \
    --prodigy_annotation_jsonl ~/drop/data/qdmr-filter/diagnostics/filternum_potential/annotations.jsonl
``` 


---
#### (deprecated since diagnostics got added) Formatting data 
To format data into something that prodigy annotation tool can read we convert DROP json to a jsonlines file using
```
python -m prodigy_annotate.drop_to_prodigy \
    --drop_json ~/drop/data/drop_wqdmr_programs-ns/drop_dataset_train.json \
    --output_jsonl ~/drop/data/drop_wqdmr_programs-ns/train_prodigy.jsonl
```