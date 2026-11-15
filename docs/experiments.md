# Matriks Eksperimen Executable

Berkas YAML di `configs/` benar-benar dijalankan oleh `graphhall benchmark`, bukan manifest deskriptif. Key asing, seed/k duplikat, tipe salah, angka nonfinite, path absolut/keluar proyek, atau input wajib yang hilang ditolak sebelum training.

## Bootstrap Clone Baru

Data/model tidak ikut Git. Jika input 10K belum ada, buat sekali dari root repositori. **Jangan menimpa input frozen yang sudah diaudit.**

```bash
uv run graphhall prepare \
  --cache-path data/raw/qa_data.jsonl \
  --output data/processed/halueval-qa-full.jsonl \
  --max-records 10000 --seed 42
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run graphhall encode \
  --input data/processed/halueval-qa-full.jsonl \
  --output data/processed/halueval-qa-full.npz \
  --model-name microsoft/deberta-v3-base \
  --revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --batch-size 4 --max-length 512 --device cpu
```

`prepare` tanpa `--source-file` memakai URL commit pinned dan merekam identitasnya. Prepared/embedding pada konfigurasi ini mempunyai stem sidecar yang sama; sesudah encoding, benchmark memverifikasi raw source dan merekonstruksi provenance source/split dengan penanda eksplisit. Gunakan stem berbeda pada konfigurasi sendiri bila ingin mempertahankan kedua sidecar secara terpisah.

## Konfigurasi Penuh 10K

`configs/qa-full.yaml` memakai `data/processed/halueval-qa-full.jsonl`, `data/processed/halueval-qa-full.npz`, dan sumber pinned `data/raw/qa_data.jsonl`. Benchmark tidak mengunduh, menyiapkan, atau meng-encode data. Sebelum training, hash raw source diperiksa, pemilihan kandidat/split seed 42 dihitung ulang, prepared-file hash dicocokkan dengan metadata embedding, serta model/revision/dimensi fitur divalidasi. Rekonstruksi metadata sumber yang hilang/bertabrakan ditandai pada provenance run.

Matriks utama berisi 13 job: GCN/GAT/GT × k `{2,5,10}`, seed 42; satu linear head; serta train-only cosine voting per k. Fitur memakai question + candidate answer, tanpa knowledge, maksimum 512 token; CPU 4 thread.

```bash
uv run graphhall benchmark --config configs/qa-full.yaml
uv run graphhall benchmark --config configs/qa-full.yaml --resume
uv run graphhall report --input-dir results/qa-full --output-dir reports/qa-full
```

Matriks multiseed mengulang learner dengan seed 42/43/44 pada **prepared rows, embeddings dan split yang sama**. Ini 39 job, bukan tiga split dataset berbeda:

```bash
uv run graphhall benchmark --config configs/qa-multiseed.yaml --resume
uv run graphhall report --input-dir results/qa-multiseed --output-dir reports/qa-multiseed
```

Matriks 39 job dan integrasi model asli sudah dijalankan serta diaudit; lihat [hasil aktual](results.md). Jangan menjalankan workload 10K/model asli dalam CI. CI hanya memakai fixture kecil CPU tanpa unduhan weights.

## Resume, Integritas, dan Replay

Setiap job berada di `results/<matrix>/runs/<learner-seed-k>/`. Completion manifest dipublikasikan atomik setelah konfigurasi efektif, provenance, graph/fitur rekonstruksi, history, checkpoint learner trainable, probabilitas prediksi dan metrik tersimpan serta di-hash. Resume hanya melewati job lengkap dengan identitas config/input/model/**source implementation dan versi dependency numerik** yang cocok serta artifact hash valid. Mismatch atau checkpoint korup ditolak; output sementara yang tidak lengkap dibuang dan dilatih ulang.

Replay memverifikasi identitas, memakai pool fitur transduktif dan graf yang terekam, memulihkan checkpoint lalu membandingkan probabilitas dengan prediksi tersimpan. **Bukan inferensi inductive untuk pertanyaan baru.** Fine-tuned replay membangun arsitektur encoder dari checkpoint dan hanya memakai tokenizer pinned dari cache lokal, tanpa mengunduh model. kNN tidak diberi checkpoint palsu; voting deterministik direplay dari fitur dan label training terverifikasi.

```bash
uv run graphhall predict --run-dir results/qa-multiseed/runs/gcn-seed42-k2
```

Report menolak run dengan identitas provenance berbeda, termasuk mismatch terhadap root `matrix.json`; artifact hash yang valid saja tidak membuktikan hasil dapat dibandingkan. Report menghasilkan `results.md` dan `results.csv`, memperlihatkan semua k/seed, sample SD beserta jumlah seed, baris terpilih melalui validasi, dan job pending. kNN deterministik tidak menghasilkan replikasi stokastik independen hanya karena dicatat dengan tiga seed.

## Baseline Fine-Tuned

`finetune` benar-benar melatih encoder DeBERTa + dropout 0,1/linear classifier pada `[CLS]`, memilih checkpoint dari validation loss. Baseline `linear` hanya melatih head pada fitur encoder beku.

```bash
uv run graphhall finetune \
  --input data/processed/halueval-qa-full.jsonl \
  --output-dir results/finetuned-deberta \
  --model-name microsoft/deberta-v3-base \
  --revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --device cpu --batch-size 1 --max-length 512 --epochs 3 \
  --learning-rate 0.00002 --gradient-accumulation 8
```

Model pinned dapat diunduh bila belum ada di cache. Tambahkan `--max-steps 2` untuk integrasi terbatas: cap dihitung sebagai **optimizer update sesudah gradient accumulation**, bukan jumlah microbatch. Checkpoint/metrik menandai `not_full_training=true`; **jangan melabeli run capped sebagai baseline penuh**. Full fine-tuning 10K belum dieksekusi; dua update model asli telah diverifikasi.

## Fixture Integrasi

`configs/qa-integration.yaml` adalah konfigurasi 30 baris fixture dan cached features 8-D, dua epoch, k=2, satu seed, tanpa fine-tuning/unduhan model. Buat fixture terlebih dahulu, misalnya melalui setup test; benchmark menolak input yang belum ada. Ini berbeda dari integrasi 60 baris dengan DeBERTa asli yang dicatat pada hasil aktual.
