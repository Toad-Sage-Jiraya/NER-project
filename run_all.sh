#!/usr/bin/env bash
# Usage: bash run_all.sh [SEEDS...]   (default seeds: 1 7 42)
set -e
SEEDS=${@:-1 7 42}
cd src
for T in DISEASE; do
  t=$(echo $T | tr 'A-Z' 'a-z')
  python prepare_data.py --data_root ../data --corpus spaccc --types $T --out ../processed_$t
  for M in bilstm bilstm_attn cnn_bilstm cnn_bilstm_attn cnn_bigru_attn; do
    for S in $SEEDS; do
      echo "=== Running $M on $t (Seed $S) ==="
      python train.py    --data ../processed_$t --model $M --seed $S --out ../runs_$t
      python evaluate.py --data ../processed_$t --run ../runs_$t/${M}_seed$S
    done
  done
  python summarize.py ../runs_$t
done
python make_report.py ..
