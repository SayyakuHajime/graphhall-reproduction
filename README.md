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
    <li><a href="#hasil-smoke">Hasil smoke yang terukur</a></li>
    <li><a href="#dokumen">Dokumen</a></li>
    <li><a href="#batasan">Batasan dan lisensi</a></li>
  </ol>
</details>

## Tentang
<a id="tentang"></a>

Repositori ini mengimplementasikan jalur lokal `prepare → encode → run` untuk tiga learner GraphHall: GCN, GAT, dan GT. Satu node merepresentasikan satu kandidat jawaban QA; encoder DeBERTa-v3-base dibekukan dan menghasilkan vektor `[CLS]` 768-dimensi.[4] Graf cosine k-NN menghubungkan sampel yang mirip. Paper melaporkan konfigurasi dan hasil pada benchmark penuh; smoke lokal di sini hanya memeriksa bahwa pipeline nyata berjalan, bukan mereplikasi atau mengesahkan angka benchmark.[1]

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
      │ encode: DeBERTa-v3-base beku, question–answer, [CLS]
      ▼
768-D embeddings + model/split provenance
      │ run: cosine k-NN → GCN / GAT / GT → test metrics
      ▼
metrics, predictions, histories, checkpoints, config, provenance
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

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Hasil smoke yang terukur
<a id="hasil-smoke"></a>

Pada run CPU yang diverifikasi di repositori ini, setiap learner memproses 60 kandidat dan 6 kandidat test. GCN memperoleh akurasi `0.50` / F1 hallucinated `0.00`, GAT `0.67` / `0.50`, dan GT `0.50` / `0.00`. Ukuran test ini terlalu kecil untuk kesimpulan kinerja; angka tersebut hanya bukti eksekusi. Lihat [catatan hasil lengkap](docs/results.md) untuk parameter count, graph stats, hash, dan batas interpretasi.

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Dokumen
<a id="dokumen"></a>

- [Arsitektur dan keputusan reproduksi](docs/architecture.md)
- [Tech stack dan lingkungan](docs/tech-stack.md)
- [Perintah reproduksi dan artefak](docs/reproduction.md)
- [Hasil smoke aktual](docs/results.md)
- [Konfigurasi smoke yang dicatat](configs/smoke.yaml)

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Batasan dan lisensi
<a id="batasan"></a>

Implementasi ini tidak mengklaim mereproduksi fine-tuning baseline DeBERTa, benchmark penuh, mode inductive, atau angka latency paper. Detail yang tidak dikunci sumber dicatat sebagai pilihan lokal di dokumen reproduksi. Tidak ada lisensi repositori yang dipilih; jangan menganggap kode atau artefak ini berlisensi untuk redistribusi.

<div align="right"><a href="#readme-top">↖ Kembali ke atas</a></div>

---

## Sources

[1] https://doi.org/10.1109/TAI.2026.3715425 — GraphHall: A Graph-Based Framework for Hallucination Detection in Large Language Models
[2] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/README.md — HaluEval README at pinned commit
[3] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json — HaluEval QA data at pinned commit
[4] https://huggingface.co/microsoft/deberta-v3-base/tree/8ccc9b6f36199bec6961081d44eb72fb3f7353f3 — DeBERTa-v3-base pinned Hub snapshot
