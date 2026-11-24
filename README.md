# MultiClinAI: BiLSTM vs CNN+BiLSTM vs CNN+BiLSTM+Attention

1. Unzip the MultiClinAI dataset into `data/` (so `data/MultiClinAI-training_data-260206/SPACCC_DisTEMIST_diseases/...` exists).
2. `python -m venv venv`, activate it, `pip install -r requirements.txt`.
3. Smoke test: `cd src && python prepare_data.py --data_root ../data/MultiClinAI-training_data-260206 --types DISEASE --out ../processed_disease && python train.py --data ../processed_disease --model cnn_bilstm_attn --epochs 1 --out ../runs_disease`
4. Full experiment: `bash run_all.sh` (or `.\run_all.ps1`). Quick version: `bash run_all.sh 42`.
5. Results: `results/results_summary.md` (generated from the real metric files).

Models: bilstm, bilstm_attn, cnn_bilstm, cnn_bilstm_attn, cnn_bigru_attn (see `src/models.py`).
See `CONTEXT.md` for history, decisions, corrections and status.
Notes: one model per entity type (merging types drops ~14% of entities); documents are split 70/15/15 by document; headline metric is strict entity-level F1.
Status: the data pipeline was tested on the real dataset; the TensorFlow scripts were not executed by the author, so the first run may need small fixes.
