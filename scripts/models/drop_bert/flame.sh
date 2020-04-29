#!/usr/bin/env

export TRAINING_DATA_FILE=./resources/data/drop_acl/merged_data/iclr20_full/sample_train.json
export VAL_DATA_FILE=./resources/data/drop_acl/merged_data/iclr20_full/sample_dev.json

export DATASET_READER="drop_reader_bert_ncomb"

# Check CONFIGFILE for environment variables to set
export GPU=0

export COUNT_FIXED=false
export AUXLOSS=true
export DENLOSS=true
export EXCLOSS=true
export QATTLOSS=true
export MMLLOSS=true
export SUPFIRST=true
export SUPEPOCHS=0

export BS=4
export DROPOUT=0.2
export SEED=1
export BEAMSIZE=2
export MAX_DECODE_STEP=14
export EPOCHS=3
export DEBUG=false
SERIALIZATION_DIR=./resources/semqa/checkpoints/test

rm -r ${SERIALIZATION_DIR}

num_seconds=210
rate=0.01

allennlp train \
  --include-package semqa -s ${SERIALIZATION_DIR} training_config/semqa/train/drop_parser_bert.jsonnet &

# Obtain the id of the python process by getting the child pid
# of the allennlp command
parent_pid=$!
sleep 60
#child_pid=$(pgrep -P ${parent_pid})

echo -e "\n\nStarting pyflame for pid: ${parent_pid}\n\n"

../pyflame/src/pyflame -p ${parent_pid} -s ${num_seconds} -r ${rate} -x > bert.prof
perl ../FlameGraph/flamegraph.pl bert.prof > bert.svg

echo "Killing child process ${parent_pid}"
kill -9 ${parent_pid}