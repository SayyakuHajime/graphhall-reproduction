# Hasil Eksekusi HaluEval-QA

## Status terverifikasi

Implementasi sekarang telah menjalankan **39 eksperimen penuh pada 10.000 kandidat** melalui runner YAML baru, bukan hanya smoke atau source awal. Ada pula **6 eksperimen integrasi nyata pada 60 kandidat**, termasuk encoder DeBERTa yang benar-benar diperbarui melalui fine-tuning. **Fine-tuning penuh 10K belum dijalankan.** Hasil ini adalah perbandingan lokal dengan asumsi eksplisit, bukan klaim reproduksi identik skor paper.

- Konfigurasi penuh: `configs/qa-multiseed.yaml`, 39 completed / 0 pending; GCN/GAT/GT × k 2/5/10 × seed 42/43/44, linear × 3 seed, serta kNN × k 2/5/10 × 3 baris seed.
- Semua seed learner memakai **split yang sama**, train/validation/test 8.000/1.000/1.000, masing-masing seimbang kelas. Satu kandidat per rekaman sumber; tidak ada pasangan jawaban dari rekaman yang sama lintas split.
- Fitur asli DeBERTa-v3-base beku `(10000, 768)`, question + candidate answer sebagai pasangan tokenizer, tanpa `knowledge`, pooling `[CLS]`, maksimum 512 token; CPU 4 thread. GT menambahkan LPE 10-D.
- Raw source revision `b7253db3cdaa0ab2c382f92b26b390109174f77e`; model revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
- Raw SHA-256 `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88`; embeddings SHA-256 `5577f6a9daf4738a75eefa70a06fad4c9d8fb7e7a3c97df72d112e397dcb8778`.
- Implementation/runtime identity seluruh 45 job: `b3606c3ea7f35cd31de9629393e399ae36af81776bfb390de2bb2e3fde6376b4`. Hash mengikat source Python paket dan versi dependency numerik; perubahan source dapat membuat resume/replay menolak artefak lama.

## Ringkasan test: pemilihan k melalui validasi

Untuk setiap keluarga learner, k dipilih dari **rata-rata F1 validasi tiga seed**, bukan test. Accuracy/F1 berikut dalam persen, mean ± **sample standard deviation** antarseed. Brier/ECE adalah mean pada skala 0–1, lebih rendah lebih baik. Binary F1 memakai hallucinated (`1`) sebagai positif; probabilitas faktual (`P(label=0)`) dipakai untuk ECE dan Brier, 10 equal-width ECE bins.

| Learner | k terpilih | Accuracy (%) | F1 hallucinated (%) | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| GCN | 2 | 90.27 ± 0.47 | 90.08 ± 0.49 | 0.0859 | 0.1021 |
| GAT | 2 | 90.40 ± 0.00 | 90.20 ± 0.01 | 0.0922 | 0.1224 |
| GT | 10 | 96.20 ± 0.20 | 96.15 ± 0.21 | 0.0317 | 0.0165 |
| linear | — | 93.60 ± 0.17 | 93.57 ± 0.18 | 0.0681 | 0.1145 |
| knn | 10 | 92.10 ± 0.00 | 91.85 ± 0.00 | 0.0653 | 0.0439 |

SD ini mengukur variasi seed training pada split tetap, **bukan confidence interval atau variasi split dataset**. kNN deterministik: tiga baris seed identik dan SD nol bukan tiga replikasi stokastik independen. Skor linear yang lebih tinggi daripada GCN/GAT pada setting ini juga harus dilaporkan, bukan disembunyikan.

Report otomatis mempertahankan seluruh 39 baris, agregasi setiap k, serta pemenang validasi per seed (berbeda dari ringkasan per keluarga di atas): [Markdown](../reports/qa-multiseed/results.md), [CSV](../reports/qa-multiseed/results.csv). AUROC, AUPRC, NLL, precision, recall, Brier, ECE dan identitas tiap eksperimen tersedia pada CSV/metrik lokal. NLL kNN dapat besar karena voting memberi probabilitas tepat nol; implementasi memakai clipping minimum float64 agar hasil finite, bukan smoothing yang dilatih.

## Fine-tuning nyata: hanya integrasi terbatas

Runner production menjalankan 6 job pada 60 kandidat (48/6/6), k=5, maksimum 2 epoch learner, serta **2 optimizer update** DeBERTa, batch 1, gradient accumulation 2, panjang 96, seed 42. Checkpoint menyatakan `not_full_training=true`, `full_training=false`, dan `max_steps_unit=optimizer_update_steps`.

Audit membandingkan parameter encoder asli `deberta.encoder.layer.0.attention.self.query_proj.weight` dari cache revision pinned dengan tensor checkpoint fine-tuned: 589.824 nilai berubah, maksimum perubahan absolut `4.023313522338867e-05`. Jadi ini bukan frozen encoder + head yang disalahnamakan fine-tuning. Checkpoint dipilih dari validation loss minimum dan berhasil direplay. Skor pada enam contoh test **tidak dipakai sebagai hasil baseline penelitian**; tabel report integrasi di `reports/real-model-integration/` hanya bukti jalur eksekusi.

## Verifikasi independen

Satu review independen menemukan dua P2: pembobotan microbatch terakhir yang pendek dan pencampuran identitas run pada report. Keduanya direproduksi dengan regression test RED, diperbaiki, lalu GREEN. Seluruh 39 + 6 job dijalankan ulang pada source final; kandidat sebelum review tetap diarsipkan. Report juga diuji langsung untuk menolak gabungan dua run nyata beridentitas berbeda sebelum publikasi output.

- **70 tests passed**, satu upstream TorchScript deprecation warning; Ruff lint/format, `uv lock --check`, CLI, dan `git diff --check` lulus.
- Seluruh artifact hash pada 39 + 6 job valid; **390.360 baris prediksi** diaudit. Semua metrik validation/test dihitung ulang dengan implementasi audit terpisah; seluruh checkpoint dipilih dari minimum validation loss yang terekam.
- **45 job direplay**, 390.360 baris; selisih probabilitas maksimum `1.1920928955078125e-07` (< toleransi `1e-6`). Termasuk graph, linear, voting dan fine-tuned encoder asli.
- Resume kedua matriks lulus tanpa melatih ulang job lengkap. Regression test menolak input/config/code identity yang berubah dan artefak korup.
- Wheel/sdist berhasil dibangun. Seluruh 12 modul wheel cocok byte-for-byte dengan source; instalasi wheel terisolasi, CLI help dan replay checkpoint aktual lulus. Dependency numerik memakai instalasi lokal yang sama: ini **bukan bukti fresh dependency installation atau remote CI**.
- Bukti audit rinci disimpan lokal di `.verification/` (tidak dilacak Git): `canonical-multiseed-audit.json`, `real-model-integration-audit.json`, `canonical-replay-audit.json`, `real-finetune-weight-audit.json`, `validation-selected-family-summary.json`, `wheel-audit.json`, dan `final-pytest.log`.

## Perintah yang telah dijalankan

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run graphhall benchmark --config configs/qa-multiseed.yaml
uv run graphhall benchmark --config configs/qa-multiseed.yaml --resume
uv run graphhall report --input-dir results/qa-multiseed --output-dir reports/qa-multiseed
uv run graphhall predict --run-dir results/qa-multiseed/runs/gcn-seed42-k2
```

Mode offline di atas memerlukan data dan model yang sudah tersedia. Lihat [bootstrap dan protokol](experiments.md) untuk clone baru. Checkpoint, data, embeddings dan audit lokal tidak ikut Git; report ringkas tidak mengandung teks dataset atau bobot model.

## Batas yang belum selesai atau belum dibuktikan

- **Baseline DeBERTa fine-tuned penuh 10K/3 epoch belum dijalankan**; tersedia implementasi dan perintahnya. Jangan menyajikan dua update integrasi sebagai baseline penuh.
- CUDA tidak diuji: Torch environment ini CPU-only. Verifikasi di atas bersifat lokal; status GitHub Actions harus diperiksa pada commit yang dipublikasikan, bukan disimpulkan dari hasil lokal.
- Graf bersifat transduktif atas pool tetap. Tidak ada inferensi inductive pertanyaan baru, pengujian dataset GraphHall lain, atau reproduksi latency paper.
- Parameter GCN lokal 1.538, GAT weighted 6.178, GT weighted 20.268. Perbedaan GAT/GT terhadap paper dan keputusan edge/LPE/sampling dicatat di [paper fidelity](paper-fidelity.md); angka lokal tidak membuktikan arsitektur privat paper identik.
- Hasil 27 run dari source awal `074a7c0` adalah bukti historis terpisah, bukan pengganti matriks baru di atas. [Smoke 60-row historis](results-smoke-historical.md) dipertahankan tanpa mengubah angkanya.
