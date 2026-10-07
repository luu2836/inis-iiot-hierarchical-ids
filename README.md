# A Leakage-Aware Study of Hierarchical Divide-and-Conquer for Class-Imbalanced Industrial IoT Intrusion Detection

Reproduction code for the paper. It evaluates a **hierarchical
divide-and-conquer** intrusion-detection framework on class-imbalanced IIoT
traffic, under a **leakage-aware evaluation protocol**, across four public
datasets and five random seeds.

The study answers three questions:

1. How much does a simple hierarchical decomposition help once common
   evaluation pitfalls (duplicate records, label-leaking features) are removed?
2. Why does it help? (frequency-ordered partitioning is the active ingredient)
3. When should one use hierarchy vs loss re-weighting?

## Datasets

The code expects the following files under `IIOT_DATA_ROOT`
(default `./data`, or set the environment variable):

| Dataset | File | Source |
|---|---|---|
| Edge-IIoTset (ML subset) | `Edge-IIoTset/ML-EdgeIIoT-dataset.csv` | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot |
| ToN-IoT (network) | `ToN-IoT/train_test_network.csv` | https://research.unsw.edu.au/projects/toniot-datasets |
| CIC-IDS2017 | `CIC-IDS2017/cic_capped.csv` | https://www.unb.ca/cic/datasets/ids-2017.html (then run `code/make_cic.py`) |
| UNSW-NB15 | `UNSW-NB15/UNSW_NB15_training-set.csv` | https://research.unsw.edu.au/projects/unsw-nb15-dataset |

`cic_capped.csv` is produced from the eight original CIC-IDS2017 daily CSVs by
`code/make_cic.py` (per-class cap at 50,000 then exact deduplication).

```
data/
├── Edge-IIoTset/ML-EdgeIIoT-dataset.csv
├── ToN-IoT/train_test_network.csv
├── CIC-IDS2017/            # 8 original CIC-IDS2017 CSVs; make_cic.py writes cic_capped.csv
└── UNSW-NB15/UNSW_NB15_training-set.csv
```

## Installation

```
pip install -r requirements.txt
```

## Reproducing the results

Run everything from the `code/` directory.

```
cd code

# 0) inspect class distributions (writes bar charts to outputs/)
python eda.py

# 1) basic baselines
python baseline.py

# 2) main results table (Table 2): flat/hierarchical x GBDT/MLP, 5 seeds
python final_table.py edge
python final_table.py ton
python final_table.py unsw
python final_table.py cic

# 3) comparison with imbalance-handling methods (Table 3)
python compare_all.py unsw
python compare_all.py edge
python compare_all.py ton
python compare_all.py cic

# 4) partition-strategy ablation (Table 6): frequency vs random vs ascending
python ablation_partition.py unsw
python ablation_partition.py edge
python ablation_partition.py ton
python ablation_partition.py cic

# 5) deep (1D-CNN) baseline (Table 4)
python dl_baseline.py edge

# 6) cascade error analysis (Table 5) and K-sensitivity (Figure 6)
python cascade_error.py cic
python k_sensitivity.py

# 7) efficiency: inference latency / parameters and training time
python nn_hier.py
python train_time.py all
```

Figures are written to the `outputs/` directory.

## Code overview

| File | Purpose |
|---|---|
| `config.py` | dataset paths (`IIOT_DATA_ROOT`), seeds |
| `baseline.py` | preprocessing + RandomForest/HistGB flat baselines |
| `hierarchical.py` | hierarchical divide-and-conquer (fixed partition) |
| `search_partition.py` | validation-driven partition search |
| `ablation_partition.py` | frequency vs random vs ascending ordering |
| `final_table.py` | main multi-seed table + significance tests |
| `compare_all.py` | comparison with re-weighting / SMOTE / focal loss |
| `dl_baseline.py` | 1D-CNN deep baseline |
| `nn_hier.py` | gated MLP instantiation + latency/parameters |
| `rare_fix.py` | tail-level refinements (negative result) |
| `per_class.py`, `plot_results.py`, `plot_compare.py`, `k_sensitivity.py`, `flow_figure.py` | figures |
| `cascade_error.py` | routing / cascade error analysis |
| `train_time.py` | wall-clock training time |
| `make_cic.py`, `edge_leak.py`, `diag_cic_dedup.py` | leakage-aware preprocessing |

## Key findings

- Hierarchical decomposition improves macro-F1 and worst-class F1, and greatly
  stabilises rare-class detection.
- Frequency-ordered partitioning is the active ingredient (random/ascending
  ordering is clearly worse).
- The optimum strategy depends on the imbalance ratio: hierarchy at moderate
  imbalance, loss re-weighting at extreme imbalance.
- Exact deduplication of CIC-IDS2017 removes a phantom +0.12 macro-F1 gain,
  motivating the leakage-aware protocol.

## Citation

```
@article{li2026hierarchical,
  title  = {A Leakage-Aware Study of Hierarchical Divide-and-Conquer for
            Class-Imbalanced Industrial IoT Intrusion Detection},
  author = {Li, Yiyan},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`).
