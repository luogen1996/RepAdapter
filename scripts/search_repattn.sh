#!/bin/sh
echo "start"
devices=$1
method='repattn'
dim=8
scale=0.1
bash scripts/exp.sh $devices $method  $dim  $scale  2>&1 | tee ./logs/repadapter-$method-$scale.log
scale=0.5
bash scripts/exp.sh $devices $method  $dim  $scale  2>&1 | tee ./logs/repadapter-$method-$scale.log
scale=1
bash scripts/exp.sh $devices $method  $dim  $scale  2>&1 | tee ./logs/repadapter-$method-$scale.log
scale=5
bash scripts/exp.sh $devices $method  $dim  $scale  2>&1 | tee ./logs/repadapter-$method-$scale.log
scale=10
bash scripts/exp.sh $devices $method  $dim  $scale  2>&1 | tee ./logs/repadapter-$method-$scale.log