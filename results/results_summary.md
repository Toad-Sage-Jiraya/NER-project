# MultiClinAI results summary (auto-generated)

All numbers are test-set scores computed by evaluate.py; values are mean +- std over seeds.
Headline metric: strict entity-level F1. Token accuracy is inflated by the many O tokens.

## Entity type: DISEASE

Data: 750 documents, 306004 tokens, 8021 entities (155 overlapping spans dropped).

| Model | Token acc | Entity P | Entity R | Entity F1 (strict) | Entity F1 (relaxed) | Params | Seeds |
|---|---|---|---|---|---|---|---|
| bilstm | 0.9456 +- 0.0012 | 0.4909 +- 0.0181 | 0.4067 +- 0.0308 | 0.4446 +- 0.0256 | 0.7602 +- 0.0063 | 1187656 | 3 |
| bilstm_attn | 0.9391 +- 0.0006 | 0.3360 +- 0.0366 | 0.2719 +- 0.0521 | 0.2995 +- 0.0440 | 0.7377 +- 0.0104 | 1451336 | 3 |
| cnn_bilstm | 0.9453 +- 0.0001 | 0.4880 +- 0.0188 | 0.3971 +- 0.0164 | 0.4379 +- 0.0174 | 0.7628 +- 0.0061 | 1263806 | 3 |
| cnn_bilstm_attn | 0.9430 +- 0.0019 | 0.4416 +- 0.0135 | 0.3255 +- 0.0907 | 0.3682 +- 0.0691 | 0.7256 +- 0.0370 | 1527486 | 3 |
| cnn_bigru_attn | 0.9434 +- 0.0016 | 0.4241 +- 0.0263 | 0.3736 +- 0.0295 | 0.3972 +- 0.0278 | 0.7513 +- 0.0080 | 1462462 | 3 |
