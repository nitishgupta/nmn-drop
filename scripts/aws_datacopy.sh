#!/usr/bin/env bash


IP=54.80.126.236
PEM=~/nitish.pem

scp -r -i ${PEM} ./resources/data/drop/* ubuntu@${IP}:/home/ubuntu/code_data/mhqa/resources/data/drop/
