#!/bin/sh
echo "start"
devices=$1
method='repblock'
dim=8
sparse_lambda=0
model_folder='models'
scale=0
bash scripts/exp.sh $devices $method  $dim  $scale $sparse_lambda $model_folder 2>&1 | tee ./logs/repadapter-$method.log