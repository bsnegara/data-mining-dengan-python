# Sumber gambar buku

Seluruh gambar di buku dibuat dengan TikZ dan pgfplots. Folder ini memuat
sumber `.tex` setiap gambar dan berkas data `.dat` yang digambar oleh
pgfplots, dikelompokkan per bab.

Berkas `.dat` dihasilkan oleh skrip di `kode/babNN/`: skrip menulis ke
`gambar/babNN/` relatif terhadap akar repositori, jadi menjalankan skrip
akan menimpa berkas di sini dengan isi yang sama.
