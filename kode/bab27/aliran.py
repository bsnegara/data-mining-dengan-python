import numpy as np
import pandas as pd

rng = np.random.default_rng(27)


def rp(x):
    return f"{x:,.0f}".replace(",", ".")


def reservoir(aliran, k, rng):                                   #<1>
    """Sampel acak k butir dari aliran tanpa tahu panjangnya."""
    sampel = []
    for i, x in enumerate(aliran):
        if i < k:
            sampel.append(x)
        else:
            j = rng.integers(0, i + 1)
            if j < k:
                sampel[j] = x
    return sampel


item = pd.read_csv("kode/data/transaksi_lengkap_2026q1.csv",     #<2>
                   usecols=["no_struk", "nilai"])
struk = item.groupby("no_struk").nilai.sum().rename("total")
struk = struk.reset_index()
contoh = reservoir(struk.itertuples(index=False), 2000, rng)
print(f"{rp(len(struk))} struk; rata-rata total sebenarnya",
      rp(struk.total.mean()))
print("rata-rata dari sampel reservoir 2.000 struk:",
      rp(np.mean([s.total for s in contoh])))

n, rata, m2 = 0, 0.0, 0.0                                        #<3>
for x in struk.total:                    # algoritme Welford
    n += 1
    delta = x - rata
    rata += delta / n
    m2 += delta * (x - rata)
print(f"rata-rata dan sd satu lintasan: {rp(rata)} dan "
      f"{rp(np.sqrt(m2 / (n - 1)))} "
      f"(pandas: {rp(struk.total.std())})")
