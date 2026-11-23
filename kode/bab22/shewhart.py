import numpy as np
import pandas as pd

b = pd.read_csv("kode/data/batch_sangrai.csv")
x = b.durasi_detik.to_numpy(float)
fase1 = x[:30]                                                   #<1>
mr = np.abs(np.diff(fase1))                                      #<2>
pusat = fase1.mean()
sigma = mr.mean() / 1.128
bka, bkb = pusat + 3 * sigma, pusat - 3 * sigma
print(f"fase I (30 batch): rata-rata {pusat:.1f} s, "
      f"sigma (MR/1,128) {sigma:.1f} s")
print(f"batas kendali: {bkb:.1f} - {bka:.1f} detik")
luar = np.where((x > bka) | (x < bkb))[0]                        #<3>
for i in luar:
    print(f"  {b.id_batch[i]}: durasi {x[i]:.0f} s"
          f"  ({b.catatan_operator[i]})")

z = (x - pusat) / sigma                                          #<4>
for i in range(8, len(z)):
    if (z[i - 8:i + 1] > 0).all() or (z[i - 8:i + 1] < 0).all():
        print(f"  aturan 9 berturut-turut sepihak: berakhir di "
              f"{b.id_batch[i]}")
        break
else:
    print("  aturan 9 berturut-turut sepihak: tidak ada sinyal")
with open("gambar/bab22/shewhart.dat", "w") as f:
    f.write("i x\n")
    for i, v in enumerate(x, 1):
        f.write(f"{i} {v:.0f}\n")
