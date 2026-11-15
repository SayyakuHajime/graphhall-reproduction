# Reproduksi dan penggunaan CLI

## Jalur end-to-end yang disarankan

Dari root repositori, perintah berikut mengunduh data dan memakai revision DeBERTa yang dipatok. `smoke` dibatasi maksimal 80 rekaman, panjang token 96, dan 20 epoch; default-nya adalah 60 rekaman, batch 4, validasi setiap 5 epoch, seed 42, dan dua thread CPU.

```bash
uv sync --extra dev
uv run graphhall smoke \
  --output-dir results/smoke-qa-60 \
  --max-records 60 \
  --model-name microsoft/deberta-v3-base \
  --revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --batch-size 4 \
  --max-length 96 \
  --k 5 \
  --threshold 0.8 \
  --learning-rate 0.005 \
  --weight-decay 0.0005 \
  --max-epochs 20 \
  --validation-interval 5 \
  --patience 3 \
  --seed 42 \
  --threads 2
```

`configs/smoke.yaml` adalah manifest yang mencatat nilai run yang diverifikasi; CLI menerima opsi melalui argumen dan tidak membaca YAML tersebut. `results/smoke-qa-60/run/config.json` adalah konfigurasi efektif yang ditulis oleh CLI.

Data publik diunduh dari raw file HaluEval-QA pada commit `b7253db3cdaa0ab2c382f92b26b390109174f77e`; URL dan SHA-256 sumber dicatat di metadata.[2][3] Model dimuat dari revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`; revision resolusi dan hash state-dict hasil pemuatan disimpan bersama embeddings.[4]

## Tahap terpisah

Tahap yang sama dapat dijalankan satu per satu untuk inspeksi artefak:

```bash
uv run graphhall prepare \
  --cache-path data/raw/qa_data.jsonl \
  --output results/smoke-qa-60/prepared/halueval-qa.jsonl \
  --max-records 60 --seed 42

uv run graphhall encode \
  --input results/smoke-qa-60/prepared/halueval-qa.jsonl \
  --output results/smoke-qa-60/embeddings/halueval-qa.npz \
  --model-name microsoft/deberta-v3-base \
  --revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --batch-size 4 --max-length 96

uv run graphhall run \
  --input results/smoke-qa-60/prepared/halueval-qa.jsonl \
  --embeddings results/smoke-qa-60/embeddings/halueval-qa.npz \
  --output-dir results/smoke-qa-60/run \
  --learners gcn,gat,gt --k 5 --threshold 0.8 \
  --learning-rate 0.005 --weight-decay 0.0005 \
  --max-epochs 20 --validation-interval 5 --patience 3 \
  --seed 42 --threads 2 --run-type smoke
```

`run` mewajibkan IDs embeddings sama persis dengan urutan kandidat prepared. Bila sidecar encoder menyatakan `prepared_jsonl_sha256` atau `row_count`, keduanya harus cocok; mismatch menghentikan run sebelum training. NPZ fitur tanpa sidecar tetap dapat dipakai untuk fixture lokal, tetapi provenance encoder akan ditandai tidak diketahui.

## Pemilihan sampel dan evaluasi

HaluEval-QA mempunyai 10.000 rekaman sumber dengan dua field jawaban per rekaman.[2][3] Implementasi mengambil 60 rekaman secara deterministik, membagi rekaman induk 80/10/10 menjadi 48/6/6, lalu memilih satu kandidat per rekaman dengan label seimbang per split. `0` adalah jawaban faktual (`right_answer`), `1` adalah jawaban hallucinated. Kebijakan ini mengikuti unit kandidat yang digunakan evaluasi QA upstream, tetapi seed/split 60 sampel adalah keputusan smoke lokal, bukan rekonstruksi split paper.[1][2]

Semua node (termasuk val/test) hadir dalam fitur dan graf untuk protokol transduktif; hanya label train dihitung dalam loss. Validation loss menentukan checkpoint terbaik. Test labels hanya dipakai untuk metrik final setelah model dipulihkan. Binary F1 menganggap label hallucinated `1` sebagai positif.

Smoke menggunakan `τ=0.8`, `k=5`, learning rate `0.005`, weight decay `0.0005`, maksimal 20 epoch, validasi setiap 5 epoch, patience 3, dan seed 42. Paper menguji 10.000 sampel, k `{2,5,10}`, dan split 8.000/1.000/1.000; jadi run smoke ini bukan hasil benchmark dan bukan evaluasi k penuh.[1]

## Artefak yang dihasilkan

Untuk `--output-dir results/smoke-qa-60`:

```text
results/smoke-qa-60/
├── prepared/
│   ├── halueval-qa.jsonl
│   └── halueval-qa.meta.json
├── embeddings/
│   ├── halueval-qa.npz
│   └── halueval-qa.meta.json
└── run/
    ├── config.json
    ├── provenance.json
    ├── metrics.json
    ├── predictions.csv
    ├── histories.json
    └── checkpoints/
        ├── gcn-k5.pt
        ├── gat-k5.pt
        └── gt-k5.pt
```

- `prepared/*.meta.json`: URL/commit/sha sumber, jumlah baris asal dan terpilih, split, label, dan kebijakan sampling.
- `embeddings/*.meta.json`: model/revision, hash state-dict, dimensi 768, max length, pooling, konteks, dan prepared-file hash.
- `config.json`: hyperparameter, split, parameter count learner, statistik graph, hash file, dan konfigurasi run.
- `metrics.json`: metrik test per learner, epoch/loss validasi terbaik, dan parameter count.
- `histories.json`: loss train dan validation pada setiap pemeriksaan.
- `predictions.csv`: prediksi semua node, dengan split dan label gold untuk audit; metrik test tetap dihitung hanya pada test mask.
- `checkpoints/*.pt`: state terbaik berdasarkan validation loss.

`data/`, `results/`, dan checkpoint lokal diabaikan Git. Output aktual bukan bagian dari clone; jalankan ulang perintah untuk membangunnya.

## Sources

[1] https://doi.org/10.1109/TAI.2026.3715425 — GraphHall: A Graph-Based Framework for Hallucination Detection in Large Language Models
[2] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/README.md — HaluEval README at pinned commit
[3] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json — HaluEval QA data at pinned commit
[4] https://huggingface.co/microsoft/deberta-v3-base/tree/8ccc9b6f36199bec6961081d44eb72fb3f7353f3 — DeBERTa-v3-base pinned Hub snapshot
