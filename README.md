# Aplikasi Watermarking Digital dengan PCA-LSB

Aplikasi web untuk melakukan watermarking pada gambar digital menggunakan kombinasi teknik PCA (Principal Component Analysis) dan LSB (Least Significant Bit).

## Fitur

- Upload gambar digital (cover image)
- Upload watermark
- Proses watermarking otomatis
- Verifikasi watermark pada gambar
- Download hasil watermarking
- Antarmuka web yang user-friendly

## Teknologi yang Digunakan

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Scikit-learn
- Pillow (PIL)

## Cara Instalasi Lokal

1. Clone repository ini
```bash
git clone [URL_REPOSITORY]
cd [NAMA_FOLDER]
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi
```bash
streamlit run app.py
```

## Persyaratan Gambar

### Gambar Utama
- Format: JPG, JPEG, PNG
- Ukuran: Harus cukup besar
- Karakteristik: Sebaiknya memiliki variasi warna yang tidak terlalu ekstrem

### Gambar Watermark
- Format: JPG, JPEG, PNG
- Ukuran: Lebih kecil dari gambar utama
- Karakteristik: Memiliki kontras yang baik

## Cara Penggunaan

1. **Mode Tambah Watermark**
   - Upload gambar utama
   - Upload gambar watermark
   - Klik "Proses Watermarking"
   - Download hasil

2. **Mode Verifikasi Watermark**
   - Upload gambar yang ingin diperiksa
   - Klik "Periksa Watermark"
   - Lihat hasil verifikasi

## Demo

Aplikasi dapat diakses secara online di: [LINK_APLIKASI]

## Kontribusi

Kontribusi selalu welcome! Silakan buat pull request atau laporkan issues.

## Lisensi

[MIT License](LICENSE) 