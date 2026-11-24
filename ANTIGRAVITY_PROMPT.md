# Prompt to paste into Google Antigravity (Agent Manager / Editor chat)

You are working in the folder of this project (MultiClinAI_CNN_BiLSTM_Attention). Goal: set up, run and evaluate clinical NER model families on the MultiClinAI dataset, then report the REAL results.

All the code genertated shall be pushed into https://github.com/Toad-Sage-Jiraya this github account. Create new repository one with name NER project and then push all the modules like a real world project here.

keep track of all updates in a seperate context file. what tasks has been done what is pending such that if I switch to new LLM it do know the context about what is happening and what is needed to be done.

## Steps (do them in order, show me the terminal output)

0. Read `CONTEXT.md` completely before doing anything. Whenever you finish a task or change a file, append a line to section 8 (date | file | change | reason | effect on results) and, if you correct an earlier mistake, add a row to section 6.
1. Put the dataset zip in `data/` and unzip it so that `data/MultiClinAI-training_data-260206/SPACCC_DisTEMIST_diseases/en/train/*.ann` exists.
2. Create a venv, `pip install -r requirements.txt`, and check `python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"`.
3. Read every file in `src/` first. Then run the data step for DISEASE, SYMPTOM, PROCEDURE (see `run_all.sh`). Expected for DISEASE: 750 documents, 8022 entities, 156 dropped overlaps. Stop and tell me if the numbers differ.
4. Smoke test: train `cnn_bilstm_attn` for 1 epoch on `processed_disease`, then run `evaluate.py` on it. If anything crashes (this code was never executed with TensorFlow), fix it with the smallest possible change and write each fix into `CHANGELOG.md` (file, line, reason).
5. Run the full experiment: models bilstm, bilstm_attn, cnn_bilstm, cnn_bilstm_attn, cnn_bigru_attn; seeds 1 7 21 42 100; types disease, symptom, procedure. If time is short, run disease first with seeds 1 7 42 and tell me what you skipped. Run long jobs in the background and log to `logs/`.
6. Run `python src/make_report.py ..` to create `results/results_summary.md`.
7. Zip the project (without `data/`, `venv/`, `runs_*/model.keras`) as `MultiClinAI_project.zip`.

## Honesty rules (very important)

- Report exactly what the scripts print. Never hard-code, round up, edit or invent any metric, and never tune anything on the test split. Use the validation split for any tuning.
- The headline metric is strict entity-level F1. Token accuracy is high mostly because most tokens are `O`; report both and say clearly which is which.
- Improving beyond the baseline is welcome if it is honest. Allowed (each as a separately named model, same splits and seeds, each documented): CRF output layer, pretrained biomedical word vectors, units, dropout, number of attention heads, learning-rate schedule, class weights, word dropout. Not allowed: changing the test set, changing labels, training on test data, dropping hard documents.
- If a model does not beat the baseline, or nothing reaches a target such as 98%, say so plainly and show the numbers. A true result is the goal.

## Final answer must contain

- a table: model x (token accuracy, entity precision, recall, F1 strict, F1 relaxed), mean +- std over seeds, per entity type
- the list of code changes from CHANGELOG.md
- what was NOT run and why
- the updated `CONTEXT.md` (sections 5-8 brought up to date)
