# Demo Gradio — Segmentasi Glioma MMSK-3D U-Net

Demo interaktif untuk sidang: pilih kasus, telusuri irisan aksial, dan
bandingkan ground truth vs prediksi model dengan overlay berwarna.

## Struktur

```
gradio_demo/
├── app.py            # UI Gradio (Blocks): dropdown, slider, dua panel gambar, Dice
├── inference.py      # muat model, inferensi 3D, caching per-kasus, hitung Dice
├── visualization.py  # render irisan -> overlay berwarna (per-label / per-region)
├── config.py         # SEMUA path & konstanta yang perlu kamu isi
├── requirements.txt
├── README.md
└── samples/          # (opsional) salin file .npz ke sini untuk deployment
```

## Yang perlu kamu isi (cukup 3 hal)

Semua di `config.py`:

1. `CHECKPOINT_PATH` — path ke `best_model.pth`. Biarkan `None` dulu kalau mau
   menguji UI tanpa model (hanya ground truth yang muncul).
2. `SAMPLE_CASES` — path ke file `.npz` kasus contoh. Sudah terisi
   `BraTS-GLI-02303-100.npz`; tambah 2 kasus lain bila mau.
3. Di `inference.py`, fungsi `_build_model()` — impor kelas `MMSK3DUNet`-mu dan
   kembalikan instance-nya (1 baris).

Verifikasi juga `MODALITY_ORDER` dan `REGION_LABELS` di `config.py` cocok dengan
preprocessing serta definisi region di Bab III/IV.

## Menjalankan

```bash
pip install -r requirements.txt   # kalau torch belum ada di environment
python app.py
```

Lalu buka URL lokal yang muncul. Untuk link sementara yang bisa dibuka dari
laptop lain saat sidang, ubah baris akhir `app.py` menjadi `demo.launch(share=True)`.

## Catatan

- **Inferensi di-precompute.** Saat startup, model dijalankan sekali per kasus,
  hasilnya disimpan di memori. Slider hanya membaca irisan dari array yang sudah
  ada — tidak ada pemanggilan model saat slider digeser, jadi mulus.
- **Format output model.** `inference.py` mengasumsikan model multiclass
  (output logits `(1,C,128,128,128)`, di-`argmax` jadi label `{0..4}`). Kalau
  model-mu multilabel (channel sigmoid WT/TC/ET), sesuaikan postprocess yang
  sudah ditandai komentar di `_predict_volume()`.
- **Deployment ke HuggingFace Spaces.** Path absolut server tidak ada di Spaces.
  Salin file `.npz` ke `samples/`, lalu ubah `SAMPLE_CASES` ke path relatif
  seperti `"samples/BraTS-GLI-02303-100.npz"`. Hati-hati ukuran file dan privasi
  data pasien sebelum mengunggah ke layanan publik.
