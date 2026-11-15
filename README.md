<a id="readme-top"></a>

<div align="center">
  <h3 align="center">GraphHall Reproduction</h3>
  <p align="center">
    Reproduksi lokal GraphHall untuk QA hallucination detection.<br/>
    Python, PyTorch Geometric, dan Transformers.
  </p>
</div>

<div align="center">
<br/>
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
<img src="https://img.shields.io/badge/PyTorch-CPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch CPU" />
<img src="https://img.shields.io/badge/Transformers-DeBERTa-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Transformers" />
<img src="https://img.shields.io/badge/Graph-PyG-3C2179?style=for-the-badge" alt="PyTorch Geometric" />
</div>

---

<details>
  <summary>Daftar Isi</summary>
  <ol>
    <li><a href="#tentang">Tentang</a></li>
    <li><a href="#alur-kerja">Alur kerja</a></li>
    <li><a href="#mulai">Mulai</a></li>
    <li><a href="#eksperimen-full">Matriks eksperimen</a></li>
    <li><a href="#hasil-smoke">Hasil yang terukur</a></li>
    <li><a href="#dokumen">Dokumen</a></li>
    <li><a href="#batasan">Batasan dan lisensi</a></li>
  </ol>
</details>

## Tentang
<a id="tentang"></a>

Repositori ini menyediakan pipeline lokal `prepare → encode → run` serta matriks eksperimen YAML yang dapat dieksekusi untuk GCN, GAT, GT, dan pembanding linear/k-NN. Satu node merepresentasikan satu kandidat jawaban QA; konfigurasi penuh memakai fitur `[CLS]` DeBERTa-v3-base beku dengan pertanyaan sebagai konteks dan jawaban kandidat sebagai urutan pasangan tokenizer—field `knowledge` dikecualikan.[4] Graf cosine k-NN bersifat transduktif atas seluruh pool. Smoke lama hanya memeriksa jalur eksekusi; tidak membuktikan reproduksi skor paper.[1]

Dataset berasal dari HaluEval-QA resmi pada commit yang dipatok; satu jawaban faktual atau hallucinated dipilih per rekaman induk agar pasangan jawaban tidak tersebar lintas split. Label `0` berarti `right_answer`, `1` berarti `hallucinated_answer`.[2][3]

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Alur kerja
<a id="alur-kerja"></a>

```text
HaluEval-QA JSONL
      │ prepare: split rekaman induk, pilih satu kandidat
      ▼
prepared JSONL + metadata/hash
      │ encode: DeBERTa-v3-base beku, pertanyaan sebagai konteks + jawaban pasangan, [CLS]
      ▼
768-D embeddings + model/split provenance
      │ run / benchmark: cosine k-NN → GCN / GAT / GT + pembanding
      ▼
metrics, predictions, probabilities, histories, checkpoints, config, provenance
      │ report: verified aggregates, validation-selected results, pending jobs
```

`smoke` menjalankan ketiga tahap secara berurutan. Nilai default dibatasi ke 60 rekaman, panjang maksimal 96 token, batch 4, 20 epoch, validasi setiap 5 epoch, dan 2 thread CPU. Hasil dan file antara ditulis ke direktori lokal yang diabaikan Git.

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Mulai
<a id="mulai"></a>

Dari root repositori, gunakan Python 3.11 dan `uv`:

```bash
uv sync --extra dev
uv run graphhall --help
uv run pytest -q
```

Jalankan smoke CPU pada 60 rekaman HaluEval-QA dan revision model yang dipatok:

```bash
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

Perintah ini mengunduh dataset publik HaluEval-QA jika dijalankan tanpa `--source-file`, lalu mengambil berkas model yang diperlukan dari Hugging Face. Jangan menjalankannya sebagai benchmark: smoke hanya menggunakan 60 dari 10.000 rekaman sumber, dan split test hanya berisi 6 kandidat. Opsi dan urutan lengkap tersedia di [panduan reproduksi](docs/reproduction.md); skema keputusan implementasi ada di [arsitektur](docs/architecture.md).

## Matriks eksperimen penuh
<a id="eksperimen-full"></a>

`configs/qa-full.yaml` dan `configs/qa-multiseed.yaml` memakai kembali `data/processed/halueval-qa-full.jsonl` serta `data/processed/halueval-qa-full.npz`; tidak menjalankan `prepare` atau `encode` dan tidak menimpa artefak tersebut. Konfigurasi mengunci sumber, split 8.000/1.000/1.000, encoder beku dengan pertanyaan sebagai konteks dan jawaban kandidat sebagai pasangan tokenizer (knowledge dikecualikan), maksimum 512 token, serta matriks GCN/GAT/GT × k `{2,5,10}` beserta baseline linear dan cosine voting. Sumber mentah lokal diverifikasi dengan SHA-256 dan dipakai untuk memeriksa ulang pilihan kandidat serta split seed 42 sebelum training.

```bash
uv run graphhall benchmark --config configs/qa-full.yaml
uv run graphhall benchmark --config configs/qa-full.yaml --resume
uv run graphhall report --input-dir results/qa-full --output-dir reports/qa-full
uv run graphhall predict --run-dir results/qa-full/runs/gcn-seed42-k2 --learner gcn
```

Konfigurasi multiseed menjalankan seed learner 42/43/44 pada JSONL dan fitur yang sama; seed tidak mengubah split. Resume hanya memakai run lengkap yang semua hash konfigurasi, input, model, dan artefaknya cocok. Report memilih konfigurasi dari metrik validasi saja; skor test ditampilkan setelah pemilihan dan tidak dipakai untuk tuning. Replay bersifat transduktif atas pool yang sama, bukan prediksi inductive untuk pertanyaan baru. Detail ada di [panduan eksperimen](docs/experiments.md).

`linear` adalah head di fitur beku, sedangkan baseline DeBERTa fine-tuned adalah encoder trainable dengan dropout dan classifier `[CLS]`. Baseline fine-tuned tersedia melalui `graphhall finetune` atau opsi konfigurasi; ia memerlukan model/tokenizer yang dipatok dan dapat mengunduh model bila belum ada di cache. Replay fine-tuned hanya memakai tokenizer lokal yang sudah tersedia.

Jalankan matriks penuh hanya dengan anggaran waktu/RAM yang sesuai. Konfigurasi default memakai CPU karena environment PyTorch proyek CPU-only; CUDA baru dapat dipilih bila instalasi Torch mendukungnya. CI memakai fitur kecil lokal dan tidak mengunduh bobot.

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Hasil yang terukur
<a id="hasil-smoke"></a>

Runner baru telah menyelesaikan **39 eksperimen pada 10.000 kandidat**, seed learner 42/43/44 pada split tetap 8.000/1.000/1.000. Dengan k dipilih melalui rata-rata F1 validasi, GT-k10 memperoleh test F1 **96,15 ± 0,21%** (mean ± sample SD tiga seed); linear 93,57 ± 0,18%, kNN-k10 91,85%, GAT-k2 90,20 ± 0,01%, dan GCN-k2 90,08 ± 0,49%. Ini hasil implementasi lokal, **bukan bukti reproduksi identik paper**.

Enam job integrasi nyata juga selesai, termasuk dua optimizer update encoder DeBERTa, checkpoint validation-selected, dan replay. **Fine-tuning penuh pada 10K belum dijalankan.** Seluruh 45 job direplay; 390.360 baris prediksi dan metrik validation/test diaudit independen. Verifikasi kode: 70 tests passed, lint/format/lock lulus. Lihat [hasil dan batas interpretasi](docs/results.md), [report otomatis](reports/qa-multiseed/results.md), serta [smoke historis](docs/results-smoke-historical.md).

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Dokumen
<a id="dokumen"></a>

- [Arsitektur dan keputusan reproduksi](docs/architecture.md)
- [Tech stack dan lingkungan](docs/tech-stack.md)
- [Perintah reproduksi dan artefak](docs/reproduction.md)
- [Hasil 39 eksperimen dan integrasi nyata](docs/results.md)
- [Smoke 60-row historis](docs/results-smoke-historical.md)
- [Konfigurasi smoke yang dicatat](configs/smoke.yaml)
- [Matriks eksperimen dan resume](docs/experiments.md)
- [Fidelity paper dan audit parameter](docs/paper-fidelity.md)

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Batasan dan lisensi
<a id="batasan"></a>

Matriks frozen-feature HaluEval-QA 39 job telah dijalankan dan diverifikasi. Baseline fine-tuned penuh, mode inductive, dan angka latency paper belum dijalankan; fine-tuning dua update adalah integrasi terbatas, bukan hasil baseline penuh. Konfigurasi dan report menjaga pemilihan berdasarkan validasi dan mencatat perbedaan implementasi secara eksplisit; lihat [batas fidelity](docs/paper-fidelity.md). Tidak ada lisensi repositori yang dipilih; jangan menganggap kode atau artefak ini berlisensi untuk redistribusi.

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Sources

[1] https://doi.org/10.1109/TAI.2026.3715425 — GraphHall: A Graph-Based Framework for Hallucination Detection in Large Language Models
[2] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/README.md — HaluEval README at pinned commit
[3] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json — HaluEval QA data at pinned commit
[4] https://huggingface.co/microsoft/deberta-v3-base/tree/8ccc9b6f36199bec6961081d44eb72fb3f7353f3 — DeBERTa-v3-base pinned Hub snapshot
