#!/bin/sh
date
hostname

eval "$(/home/seyang/Installs/anaconda37/bin/conda shell.bash hook)"
python --version

SCRIPT=/home/seyang/Projects/IsingModel/wolff.py
OUT_DIR=/home/seyang/Projects/IsingModel/data/

python ${SCRIPT} \
    --lattice-size ${1} \
    --temperature-scale ${2} \
    --out-dir ${OUT_DIR} 
