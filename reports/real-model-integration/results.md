# Generated experiment report

Rows are read from hash-verified completed artifacts; pending matrix jobs are shown separately.
Validation-selected winners use validation metrics only. The test score is not used for selection.
Test metrics are exposed only after that validation-only selection.

## Per-seed completed experiments

| Experiment | Learner | Seed | k | Val F1 | Test F1 |
|---|---|---:|---:|---:|---:|
| finetuned-deberta-seed42 | finetuned_deberta | 42 | None | 0.6666666666666666 | 0.6666666666666666 |
| gat-seed42-k5 | gat | 42 | 5 | 0.0 | 0.0 |
| gcn-seed42-k5 | gcn | 42 | 5 | 0.0 | 0.0 |
| gt-seed42-k5 | gt | 42 | 5 | 0.0 | 0.0 |
| knn-seed42-k5 | knn | 42 | 5 | 0.8571428571428571 | 0.8571428571428571 |
| linear-seed42 | linear | 42 | None | 0.0 | 0.0 |

## Aggregate test metrics (mean ± sample standard deviation)

| Learner | k | Seed count | Accuracy | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| finetuned_deberta |  | 1 | 0.5 (SD n/a) | 0.6666666666666666 (SD n/a) | 0.33333333333333337 (SD n/a) | 0.5 (SD n/a) |
| gat | 5 | 1 | 0.5 (SD n/a) | 0.0 (SD n/a) | 0.888888888888889 (SD n/a) | 0.9166666666666665 (SD n/a) |
| gcn | 5 | 1 | 0.5 (SD n/a) | 0.0 (SD n/a) | 0.6666666666666667 (SD n/a) | 0.6388888888888888 (SD n/a) |
| gt | 5 | 1 | 0.5 (SD n/a) | 0.0 (SD n/a) | 1.0 (SD n/a) | 1.0 (SD n/a) |
| knn | 5 | 1 | 0.8333333333333334 (SD n/a) | 0.8571428571428571 (SD n/a) | 0.9444444444444444 (SD n/a) | 0.9166666666666666 (SD n/a) |
| linear |  | 1 | 0.5 (SD n/a) | 0.0 (SD n/a) | 1.0 (SD n/a) | 1.0 (SD n/a) |

## Validation-selected results

| Seed | Winner | Validation metric/value | Test F1 (post-selection) |
|---:|---|---:|---:|
| 42 | knn-seed42-k5 | f1=0.8571428571428571 | 0.8571428571428571 |

## Fidelity caveats
This is an executable local comparison, not proof that paper scores were reproduced.
The frozen encoder pairs the question context with the candidate answer; knowledge is excluded.
Evaluation is transductive over the fixed pool.
Sampling, attention-edge semantics, and some GT layer details are underspecified.
Audited vanilla PyG GAT/GT parameter counts differ from the manuscript counts; no dummy parameters are added.
The project docs/paper-fidelity.md records the measured variants and sources.

## Pending experiments

| Experiment | Learner | Seed | k |
|---|---|---:|---:|

No pending rows means the recorded matrix has no unfinished jobs.
