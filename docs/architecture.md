# Arsitektur dan pilihan reproduksi

Implementasi ini memisahkan pemilihan kandidat, encoding teks, pembentukan graf, dan pelatihan learner. Pipeline `prepare → encode → run` tetap tersedia; konfigurasi matriks `benchmark` menambah baseline, resume, replay checkpoint, dan report terverifikasi tanpa mengubah jalur lama.

## Unit data dan label

HaluEval-QA menyimpan `knowledge`, `question`, `right_answer`, dan `hallucinated_answer`. Satu rekaman upstream menyediakan dua kandidat jawaban, bukan dua unit independen.[2][3] Implementasi melakukan sampling dan split berdasarkan rekaman induk terlebih dahulu, lalu memilih tepat satu kandidat per rekaman. Untuk setiap split, pemilihan label diimbangkan secara deterministik dengan seed. Dengan demikian, satu node graf merepresentasikan satu pasangan QA–jawaban; tidak ada dua kandidat saudara yang tersebar pada split berbeda.

Pemetaan label tetap eksplisit: `0 = right_answer` dan `1 = hallucinated_answer`. Metrik precision, recall, dan F1 memakai label positif `1`. Metadata menyimpan jumlah rekaman sumber, jumlah rekaman terpilih, jumlah kandidat, distribusi split/label, seed, URL/commit sumber, serta SHA-256.

## Encoder

- Model: `microsoft/deberta-v3-base`, revision Hugging Face immutable `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.[4]
- Encoder dibekukan (`eval`, `requires_grad=False`, `no_grad`) dan dijalankan di CPU.
- Input context default adalah pertanyaan, dipasangkan dengan kandidat jawaban sebagai urutan kedua tokenizer; field `knowledge` tidak disertakan. `--include-knowledge` menambahkan `Knowledge: ...` pada konteks dan merupakan varian lokal, bukan default.
- Tokenisasi memakai batch berukuran 4 dan fitur hidden state layer terakhir pada posisi `[CLS]`, berukuran 768. Smoke dibatasi 96 token; konfigurasi QA penuh memakai maksimum 512 token.
- Tokenizer SentencePiece lambat (`use_fast=False`) dipakai langsung; jalur ini menghindari konversi ke tokenizer cepat dan kebutuhan converter protobuf opsional.
- Metadata embeddings mencatat prepared-file hash, model, revision yang diminta dan terselesaikan, hash isi state-dict model, kebijakan konteks, pooling, dimensi, batch, dan panjang token.

Paper menyebut DeBERTa-v3-base dan representasi `[CLS]`, tetapi tidak mengunci dengan tegas apakah encoder GraphHall dibekukan atau dilatih bersama learner.[1] Pembekuan adalah keputusan implementasi yang belum dikunci paper. Input question-only mengikuti uraian QA paper (§V.C.3, hlm. 10), sedangkan penggunaan field knowledge merupakan varian tambahan.

## Graf sampel

1. Normalisasi embedding L2, hitung cosine k-NN per node dalam chunk.
2. Pilih sampai `k` tetangga terdekat selain node itu sendiri; pertahankan edge bila cosine `>= τ` (bukan `>`).
3. Edge berarah dari neighbor ke query: baris pertama `edge_index` berisi sumber tetangga, baris kedua berisi query tujuan. Graf utama tidak disimetriskan.
4. Tambahkan self-loop eksplisit berbobot `1` pada forward pass. Tidak ada self-loop yang dihitung sebagai k-neighbor.
5. GCN menerima bobot cosine skalar sebagai `edge_weight`. GAT dan GT menerima nilai yang sama sebagai atribut edge satu-dimensi (`edge_dim=1`).[1]

Bobot cosine yang dilaporkan paper jelas untuk graph edges, tetapi cara memasok bobot ke setiap layer attention tidak ditentukan secara operasional.[1] Penggunaan scalar edge attributes di GAT/GT adalah pilihan reproduksi ini dan parameter count aktual dilaporkan tanpa dipaksa agar sama dengan tabel paper.

## Positional encoding dan learner

GT menambahkan 10 eigenvector non-trivial dari normalized Laplacian ke 768 dimensi encoder, sehingga input GT berukuran 778. Untuk LapPE saja, adjacency directed disimetriskan dengan `max(A, Aᵀ)`; eigenvalue nol (termasuk mode komponen disconnected) dikecualikan, tanda eigenvector distabilkan berdasarkan elemen dengan magnitudo terbesar, lalu kolom yang kurang dipad dengan nol. Graf GNN tetap directed.

| Learner | Lapisan | Konfigurasi efektif | Parameter learner terukur |
|---|---|---|---:|
| GCN | `GCNConv` | 768 → 2; bobot cosine scalar; self-loop dikelola eksplisit | 1.538 |
| GAT | `GATConv` | 768 → 2; 4 head, rata-rata head (`concat=False`), `edge_dim=1` | 6.178 |
| GT | `TransformerConv` | 778 → 2; 4 head, rata-rata head, `beta=True`, dropout 0,1, `edge_dim=1`, root weight | 20.268 |

Parameter count di tabel adalah parameter learner trainable, tidak termasuk encoder DeBERTa yang dibekukan. Paper melaporkan 1.538 / 6.158 / 38.442 untuk learner GCN/GAT/GT.[1] GCN cocok pada konfigurasi ini; GAT dan GT aktual tidak cocok dengan angka paper. Detail opsi edge/head/skip serta versi pustaka tidak cukup terkunci di artikel untuk menyatakan bahwa selisih tersebut punya satu penyebab pasti.

## Pelatihan dan kebocoran label

Training memakai fitur dan topologi seluruh node (protokol transduktif); loss cross-entropy berbobot kelas dihitung hanya pada train mask. Validation labels memilih checkpoint dengan loss minimum dan early stopping. Test labels tidak memengaruhi gradien atau pemilihan checkpoint; test metrics dihitung setelah checkpoint terbaik dimuat. Fitur test tetap terlihat pada graph message passing, sesuai interpretasi full-graph transduktif, bukan evaluasi inductive.[1]

Smoke memakai `τ=0.8`, `k=5`, learning rate `0.005`, weight decay `0.0005`, maksimal 20 epoch, validasi setiap 5 epoch, patience 3, dan seed 42. Konfigurasi benchmark penuh tersedia terpisah; paper menggunakan learning rate `0.005`, weight decay `0.0005`, threshold `0.8`, dan menguji `k ∈ {2,5,10}`; jadwal validasi artikel memiliki ketidakselarasan antara prosa dan algoritme.[1]

## Matriks, pembanding, dan provenance

`configs/qa-full.yaml` memakai fitur 10K yang sudah dikodekan; `qa-multiseed.yaml` menggunakan split yang sama untuk seed learner 42/43/44. Matriks membangun graph/Laplacian encoding sekali per k, lalu menjalankan GCN/GAT/GT, frozen-feature `linear`, dan cosine kNN yang hanya memakai label train. Fine-tuned DeBERTa adalah classifier tersendiri yang dapat diaktifkan; ia bukan alias untuk baseline linear.

Setiap run menyimpan hash sumber/prepared/features/model/config, graph dan LPE yang digunakan, checkpoint terbaik, history, probabilitas prediksi semua node, serta metrik validasi/test. Resume hanya melewati artefak lengkap dengan semua hash cocok. Replay memverifikasi checkpoint pada pool transduktif yang sama; ia bukan inferensi untuk node baru. `report` membuat CSV/Markdown dari manifest dan metrik ber-hash, memisahkan hasil per seed, agregat, hasil pilihan validasi, dan job yang masih pending. Detail konfigurasi dan CLI ada di [panduan eksperimen](experiments.md).

## Sources

[1] https://doi.org/10.1109/TAI.2026.3715425 — GraphHall: A Graph-Based Framework for Hallucination Detection in Large Language Models
[2] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/README.md — HaluEval README at pinned commit
[3] https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json — HaluEval QA data at pinned commit
[4] https://huggingface.co/microsoft/deberta-v3-base/tree/8ccc9b6f36199bec6961081d44eb72fb3f7353f3 — DeBERTa-v3-base pinned Hub snapshot
