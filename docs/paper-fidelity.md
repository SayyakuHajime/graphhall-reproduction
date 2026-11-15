# Fidelity Paper dan Asumsi Implementasi

Dokumen ini memisahkan metode yang diimplementasikan dari detail yang tidak dapat dipastikan melalui accepted manuscript. **Bukan pernyataan bahwa angka paper sudah direproduksi.**

## Protokol Reproduksi

| Aspek | Manuscript | Implementasi Lokal | Batas Fidelity |
|---|---|---|---|
| Data QA | HaluEval-QA, 10.000 rekaman | Satu kandidat per rekaman; split train/validation/test 8.000/1.000/1.000, kelas seimbang | Seed split 42 dan pemilihan kandidat adalah kontrol lokal eksplisit; identitas sampel tidak diklaim sama dengan paper. |
| Representasi | Kandidat jawaban yang dikondisikan pertanyaan | `microsoft/deberta-v3-base` frozen; question sebagai sequence pertama, candidate answer sebagai pasangan; maksimum 512 token, last-layer `[CLS]` | Field `knowledge` dikecualikan. |
| Graf | Cosine k-nearest-neighbor | Edge terarah neighbor→query; threshold 0,8; k `{2,5,10}`; transduktif atas seluruh pool | Arah edge, perbandingan threshold, self-loop dan simetrisasi LPE adalah keputusan code-level; tidak diklaim rekonstruksi identik. |
| Learner | GCN, GAT, GT | Satu layer GCN/GAT/TransformerConv, dua logit; GT menambahkan LPE 10-D | Default layer PyG dan profile `weighted`/`topology` dicatat pada konfigurasi efektif. |
| Training | Maksimum 400 epoch, validasi setiap 20, patience 3, learning rate 0,005, weight decay 0,0005 | Default mengikuti setting tersebut; checkpoint minimum validation loss | Seed learner tidak mengubah split prepared. Label test tidak dipakai untuk fit, stopping atau pemilihan konfigurasi. |
| Baseline | Pembanding encoder frozen dan fine-tuned | Linear head pada fitur beku; cosine voting hanya label training; encoder DeBERTa trainable + dropout/linear classifier opsional | Frozen-feature head bukan fine-tuning. Implementasi lengkap tidak berarti semua baseline penuh sudah dilatih. |

## Audit Parameter

Audit terbatas pada source awal `074a7c0` menghitung parameter aktual varian layer PyG. GCN cocok dengan angka manuscript; GAT/GT yang diaudit tidak cocok:

- GCN: **1.538** parameter, cocok dengan Table X.
- GAT input 768/output 2, empat mean heads: 6.162 tanpa edge feature dengan bias, 6.160 tanpa edge feature/bias, 6.178 dengan edge feature 1-D dan bias, 6.176 dengan edge feature tanpa bias. Angka paper: **6.158**.
- GT input 778/output 2, empat mean heads: 20.254 tanpa edge feature/beta; 20.260 tanpa edge feature dengan beta; 20.262 dengan edge feature tanpa beta; 20.268 dengan edge feature/beta. Angka paper: **38.442**.
- Tidak ada varian vanilla GAT/GT yang diuji tersebut yang cocok. Tidak ada dummy parameter atau dimensi tersembunyi yang ditambahkan untuk memaksa kecocokan.

Audit ini menjelaskan implementasi yang dihitung, **bukan bukti arsitektur privat paper**. Bukti lokal `.verification/paper-parameter-audit.json` berasal dari snapshot source awal immutable. Parameter aktual learner saat ini tetap dihitung dari modul yang dieksekusi dan disimpan pada checkpoint.

## Pemilihan dan Pelaporan

Validation loss memilih checkpoint. Report memakai metrik validasi untuk baris pemenang dan tidak memilih seed/learner/konfigurasi berdasarkan skor test. Seluruh hasil sweep, mean ± sample SD dan jumlah seed tetap terlihat; job pending tidak disamarkan sebagai selesai. Semua seed memakai split yang sama, sehingga SD bukan confidence interval atau variasi dataset split. Repetisi kNN deterministik tidak menghasilkan eksperimen stokastik independen.

Smoke/integrasi bukan benchmark penuh. Fine-tuning yang dibatasi `max_steps` secara eksplisit ditandai bukan full training. [Hasil aktual](results.md) memisahkan matriks frozen-feature 10K yang sudah selesai dari fine-tuning penuh 10K yang belum dijalankan.

## Sumber

- GraphHall accepted manuscript, §IV dan Tables I, II, X: https://doi.org/10.1109/TAI.2026.3715425
- HaluEval QA commit `b7253db3cdaa0ab2c382f92b26b390109174f77e`: https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json
- DeBERTa-v3-base revision pinned: https://huggingface.co/microsoft/deberta-v3-base/tree/8ccc9b6f36199bec6961081d44eb72fb3f7353f3
