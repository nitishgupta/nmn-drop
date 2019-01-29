#!/usr/bin/env

SERIALIZATION_DIR=/home/ngupta19/code/mhqa/pycode/resources/semqa/checkpoints/hotpotqa_bool_wosame/sample_parser/BS_4/LR_0.001/Drop_0.2/BeamSize_32/MaxDecodeStep_12/
WEIGHTS_TH=best.th

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