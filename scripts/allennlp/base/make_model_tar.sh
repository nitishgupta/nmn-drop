#!/usr/bin/env

SERIALIZATION_DIR=./resources/semqa/checkpoints/drop/date_num/date_num_100/drop_parser/BS_8/LR_0.001/Drop_0.2/TOKENS_qanet/ED_100/RG_1e-4/GACT_false/GPROGS_false/QPSIMKEY_raw-enc/QAL_true/EXL_true/QATL_true/MML_true/SUPFIRST_true/SUPEPOCHS_10/S_100/test/
# WEIGHTS_TH=best.th
WEIGHTS_TH=model_state_epoch_43.th

MODEL_ARCHIVE=${SERIALIZATION_DIR}/model.tar.gz

if [ -f "${MODEL_ARCHIVE}" ]; then
    echo "Model archive exists: ${MODEL_ARCHIVE}"
    exit 1
fi

echo "Making model.tar.gz in ${SERIALIZATION_DIR}"

cd ${SERIALIZATION_DIR}

if [ -d "modeltar" ]; then
    echo "modeltar directory exists. Deleting ... "
    rm -r modeltar
fi


mkdir modeltar
cp -r vocabulary modeltar/
cp ${WEIGHTS_TH} modeltar/weights.th
cp config.json modeltar/
cd modeltar
tar -czvf model.tar.gz ./*
cp model.tar.gz ../
cd ../
rm -r modeltar