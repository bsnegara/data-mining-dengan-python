import numpy as np
import pandas as pd

data = pd.DataFrame({
    "kadar_air": ["rendah"] * 4 + ["sedang"] * 4 + ["tinggi"] * 4,
    "cacat": ["sedikit", "sedikit", "sedikit", "banyak",
              "sedikit", "sedikit", "banyak", "banyak",
              "sedikit", "sedikit", "banyak", "banyak"],
    "proses": ["GB", "FW", "FW", "GB", "GB", "FW",
               "GB", "FW", "FW", "GB", "GB", "FW"],
    "aroma": ["kuat", "kuat", "lemah", "kuat", "kuat", "lemah",
              "kuat", "lemah", "kuat", "kuat", "lemah", "kuat"],
    "keputusan": ["terima"] * 3 + ["tolak"] + ["terima"] * 2
                 + ["tolak"] * 6,
})

def entropi(y):                                                  #<1>
    p = y.value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())

def gain(df, atribut, target="keputusan"):                       #<2>
    bobot = df[atribut].value_counts(normalize=True)
    sisa = sum(bobot[v] * entropi(df.loc[df[atribut] == v, target])
               for v in bobot.index)
    return entropi(df[target]) - sisa

def id3(df, atribut, kedalaman=0):                               #<3>
    y = df["keputusan"]
    if y.nunique() == 1 or not atribut:                          #<4>
        print("    " * kedalaman + f"-> {y.mode()[0]} ({len(y)})")
        return
    g = {a: gain(df, a) for a in atribut}
    terbaik = max(g, key=g.get)                                  #<5>
    print("    " * kedalaman + "gain: " +
          ", ".join(f"{a}={v:.4f}" for a, v in g.items()))
    for v, bagian in df.groupby(terbaik):
        print("    " * kedalaman + f"[{terbaik} = {v}]")
        id3(bagian, [a for a in atribut if a != terbaik], kedalaman + 1)

print(f"H(S) = {entropi(data['keputusan']):.4f}")
id3(data, ["kadar_air", "cacat", "proses", "aroma"])
