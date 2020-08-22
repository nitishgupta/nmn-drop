#!/usr/bin/env

SERIALIZATION_DIR=./resources/checkpoints/drop-w-qdmr/qdmr-v2_iclrfull-v2-ss-cnt/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_3_HEM_3_BM_1/S_42_PreBIO_SqNMN
WEIGHTS_TH=best.th

MODEL_ARCHIVE=${SERIALIZATION_DIR}/model.tar.gz

if [ -f "${MODEL_ARCHIVE}" ]; then
    echo "Model archive exists: ${MODEL_ARCHIVE}"
    exit 1
fi

echo "Making model.tar.gz in ${MODEL_ARCHIVE}"

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