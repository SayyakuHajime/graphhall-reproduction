# Hasil run aktual: HaluEval-QA smoke

**Status: berhasil dijalankan dengan DeBERTa-v3-base asli dan seluruh learner GCN/GAT/GT.** Ini smoke CPU 60 rekaman—bukan benchmark paper dan bukan klaim reproduksi performa.

## Perintah yang dijalankan

```bash
.venv/bin/graphhall smoke \
  --output-dir results/smoke-qa-60 \
  --max-records 60 \
  --model-name microsoft/deberta-v3-base \
  --revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --batch-size 4 --max-length 96 \
  --k 5 --threshold 0.8 \
  --learning-rate 0.005 --weight-decay 0.0005 \
  --max-epochs 20 --validation-interval 5 --patience 3 \
  --seed 42 --threads 2
```

## Data, model, dan graf terukur

- Sumber: HaluEval-QA pinned commit `b7253db3cdaa0ab2c382f92b26b390109174f77e`, 10.000 rekaman sumber, SHA-256 `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88`.[2][3]
- Sampel: 60 rekaman induk, 60 kandidat. Split train/val/test `48/6/6`; setiap split seimbang label (`24/24`, `3/3`, `3/3`). Satu kandidat dipilih per rekaman, dengan label `0=faktual`, `1=hallucinated`.
- Encoder: `microsoft/deberta-v3-base`, revision diminta dan terselesaikan `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, hidden size 768, question-only, frozen, pooling `[CLS]`.[4]
- Hash state-dict DeBERTa yang dimuat: `b7d2d5b6bb59fe74c90d4406b41e9b8e5e42e5819fd750547a880c0315db2ad4`.
- Prepared JSONL SHA-256: `295b1c0b8da20ea209cc44735e31497469f233f30e6e51c6708d9467974a9395`.
- Embeddings: `(60, 768)`, seluruh nilai finite; NPZ SHA-256 `629230aa991c05bbbbb186c2d7a8121d24038696b4252768a16652da15eac2f3`.
- Graf: 300 directed k-NN edges (5 per query; semua melewati `τ=0.8`), 60 self-loop eksplisit, total 360 edge setelah self-loop, 0 query tanpa neighbor. Bobot edge terukur `0.998396–0.999711`.

## Metrik test

| Learner | Parameter learner | Epoch terbaik | Validation loss | Accuracy | Precision | Recall | F1 (positif=hallucinated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCN | 1.538 | 20 | 0,650654 | 0,5000 | 0,0000 | 0,0000 | 0,0000 |
| GAT | 6.178 | 20 | 0,630726 | 0,6667 | 1,0000 | 0,3333 | 0,5000 |
| GT | 20.268 | 20 | 0,609999 | 0,5000 | 0,0000 | 0,0000 | 0,0000 |

Parameter count di atas hanya learner trainable, bukan parameter DeBERTa frozen. Tabel I paper melaporkan `1.538 / 6.158 / 38.442` untuk GCN/GAT/GT.[1] GCN cocok, tetapi GAT dan GT aktual berbeda untuk konfigurasi `edge_dim=1`/`TransformerConv` yang eksplisit; implementasi tidak memaksa angka paper.

## Batas interpretasi

Test berisi enam contoh seimbang. Satu perubahan prediksi mengubah recall/F1 secara besar; nilai ini tidak cukup untuk menyimpulkan learner terbaik. Tidak ada baseline fine-tuned DeBERTa atau voting k-NN yang direproduksi. Run ini membuktikan tahapan persiapan data, encoding model asli, konstruksi graf, training, checkpoint, dan ekspor metrik berjalan dari awal sampai akhir.

## Artefak lokal

Output berada di `results/smoke-qa-60/` dan tidak dilacak Git:

- `prepared/halueval-qa.jsonl` + metadata sumber/split
- `embeddings/halueval-qa.npz` + metadata model/revision/hash
- `run/config.json`, `run/provenance.json`, `run/metrics.json`
- `run/predictions.csv` (180 baris: 60 node × 3 learner)
- `run/histories.json` (empat pemeriksaan validasi/learner pada epoch 5, 10, 15, 20)
- `run/checkpoints/{gcn,gat,gt}-k5.pt`

Konfigurasi manifest yang mencatat run ini ada di [`configs/smoke.yaml`](../configs/smoke.yaml); file `run/config.json` adalah konfigurasi efektif yang ditulis CLI.

## Sources

[1] https://doi.org/10.1109/TAI.2026.3715425 — GraphHall: A Graph-Based Framework for Hallucination Detection in Large Language Models
[2] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/README.md — HaluEval README at pinned commit
[3] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json — HaluEval QA data at pinned commit
[4] https://huggingface.co/microsoft/deberta-v3-base/tree/8ccc9b6f36199bec6961081d44eb72fb3f7353f3 — DeBERTa-v3-base pinned Hub snapshot
