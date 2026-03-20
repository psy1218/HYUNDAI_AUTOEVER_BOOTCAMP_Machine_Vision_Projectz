#!/bin/bash

cd /home/psy1218/projects/1_pro
mkdir -p logs

echo "===== EXP1 START ====="
python exp1_baseline_simple.py | tee logs/exp1_baseline_simple.log
echo "===== EXP1 END ====="

echo "===== EXP2 START ====="
python exp2_focal_simple.py | tee logs/exp2_focal_simple.log
echo "===== EXP2 END ====="

echo "===== EXP3 START ====="
python exp3_bigimg_simple.py | tee logs/exp3_bigimg_simple.log
echo "===== EXP3 END ====="

echo "===== EXP4 START ====="
python exp4_augweak_simple.py | tee logs/exp4_augweak_simple.log
echo "===== EXP4 END ====="

echo "===== ALL EXPERIMENTS DONE ====="
