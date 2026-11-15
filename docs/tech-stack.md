# Tech stack dan lingkungan

## Runtime

| Komponen | Dukungan/versi yang dipatok | Peran |
|---|---|---|
| Python | `>=3.11,<3.12` | Runtime aplikasi dan CLI |
| uv | `uv.lock` | Resolusi, sync, dan instalasi deterministik |
| PyTorch | `>=2.6,<3`, sumber wheel CPU eksplisit di `pyproject.toml` | Model dan training CPU |
| PyTorch Geometric | `>=2.6,<3` | `GCNConv`, `GATConv`, `TransformerConv` |
| Transformers | `>=4.48,<5` | DeBERTa-v3-base dan tokenizer |
| SentencePiece | `>=0.2,<1` | Tokenizer DeBERTa slow |
| NumPy / SciPy | `numpy>=1.26,<3`, `scipy>=1.12,<2` | Array, k-NN graph support, eigenvectors |
| scikit-learn | `>=1.4,<2` | Accuracy, precision, recall, binary F1 |
| pytest / Ruff | dev extra: `pytest>=8,<9`, `ruff>=0.9,<1` | Tes, lint, format |

Versi konkret yang terpasang dikunci di `uv.lock`. Runtime smoke yang tercatat menggunakan PyTorch CPU dan dua thread; tidak memerlukan CUDA, akun layanan, API berbayar, atau dashboard. `uv sync --extra dev` memasang proyek beserta dependensi dev.

## Artefak eksternal

- Dataset HaluEval-QA diambil dari raw GitHub pada commit `b7253db3cdaa0ab2c382f92b26b390109174f77e`; file berisi 10.000 baris JSONL dan SHA-256 `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88`.[2][3]
- `microsoft/deberta-v3-base` diambil pada revision immutable `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.[4]
- File model disimpan pada cache lokal Hugging Face; data, hasil, checkpoint, dan cache tidak menjadi bagian dari repositori. `.gitignore` mengecualikan `data/`, `results/`, dan `checkpoints/`.

Tidak ada lisensi repositori yang dipilih. Lisensi model dan dataset tetap mengikuti sumber upstream masing-masing; dokumen ini bukan pemberian hak redistribusi.

## Perintah pengembangan

Dari root repositori:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

## Sources

[2] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/README.md — HaluEval README at pinned commit
[3] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json — HaluEval QA data at pinned commit
[4] https://huggingface.co/microsoft/deberta-v3-base/tree/8ccc9b6f36199bec6961081d44eb72fb3f7353f3 — DeBERTa-v3-base pinned Hub snapshot
