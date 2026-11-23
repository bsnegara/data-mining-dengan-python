# Data Mining dengan Python

Kode, data, dan notebook pendamping buku *Data Mining dengan Python:
Konsep, Algoritme, dan Studi Kasus dengan CRISP-DM* (Nalar Press, 2027)
karya Benny Sukma Negara.

Seluruh contoh di buku mengikuti satu cerita: Kopi Nusantara, jaringan
kedai kopi rekaan, yang analis datanya memakai CRISP-DM untuk menjawab
persoalan bisnis dari bab ke bab.

## Struktur repositori

```
data-mining-dengan-python/
  kode/
    requirements.txt         pustaka dan versi yang dipakai di buku
    requirements-colab.txt   pustaka tambahan yang belum ada di Google Colab
    data/                    dataset Kopi Nusantara (CSV) dan skrip pembangkitnya
    bab01/ ... bab28/        skrip per bab, identik dengan listing di buku
    notebook/                notebook Google Colab per bab
  README.md
  LICENSE
```

Berkas `.out` di sebelah setiap skrip adalah keluaran yang dicetak di buku.
Bila keluaran di komputer Anda berbeda, bandingkan dengan berkas itu untuk
mengetahui apakah perbedaannya berasal dari versi pustaka. Skrip berawalan
`_` (misalnya `_ekspor_gambar.py`) hanya membuat bahan gambar buku dan
tidak dimuat sebagai listing.

## Menjalankan

Semua skrip dijalankan dari **akar repositori**, karena data dibaca dari
jalur relatif `kode/data/`:

```bash
git clone https://github.com/bsnegara/data-mining-dengan-python
cd data-mining-dengan-python
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r kode/requirements.txt
python kode/bab08/naive_bayes.py
```

Kode diuji dengan Python 3.13. Versi pustaka yang lebih baru biasanya juga
berjalan, tetapi angka keluaran dapat sedikit berbeda.

Di Google Colab, buka notebook dari folder `kode/notebook/`; sel pertama
menyalin repositori ini dan memasang pustaka yang diperlukan.

## Dataset

Seluruh data adalah rekaan dengan benih acak tetap. Kamus data lengkap,
bab pemakai, dan pola yang sengaja ditanam ada di Lampiran D buku.

| Berkas | Baris | Bab |
|---|---:|---|
| gerai.csv, menu.csv | 40, 22 | 3, 10, 16, 18 |
| transaksi_kasir_2026q1.csv, pesanan_aplikasi_2026q1.csv | 150.951, 60.500 | 3 |
| transaksi_gabungan_2026q1.csv, transaksi_lengkap_2026q1.csv | 211.451 | 3, 5, 17, 27 |
| pelanggan.csv | 8.000 | 4, 5, 7, 8, 10–13, 15, 27, 28 |
| pelanggan_mentah.csv, pelanggan_bersih.csv | 8.064, 8.000 | 4, 5 |
| pelanggan_2026q3.csv | 8.000 | 27, 28 |
| riwayat_aplikasi_2025.csv | 121.166 | 18 |
| mutu_biji.csv | 1.500 | 6, 9, 11, 14, 19, 20 |
| profil_rasa.csv, kemiripan_panel.csv | 24, 276 | 20, 21 |
| penjualan_harian.csv | 40.798 | 10, 16, 23 |
| batch_sangrai.csv, sensor_sangrai.csv | 80, 54.722 | 22, 26 |
| audit_kasir.csv | 7.679 | 22 |
| sesi_aplikasi.csv | 33.174 | 24 |
| ulasan.csv | 3.000 | 25 |

Membangun ulang seluruh data dari nol (dari akar repositori):

```bash
python kode/data/bangkitkan_data.py kode/data
python kode/bab03/seragamkan.py
python kode/bab03/gabung_dimensi.py
python kode/data/bangkitkan_data_lanjut.py kode/data
python kode/data/bangkitkan_data_rekomendasi.py kode/data
python kode/data/bangkitkan_data_mds.py kode/data
python kode/data/bangkitkan_data_proyek.py kode/data
python kode/bab05/bersihkan.py
```

## Lisensi

Kode dan data dirilis dengan lisensi MIT (lihat `LICENSE`).
