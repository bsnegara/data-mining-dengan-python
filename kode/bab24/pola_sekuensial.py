from collections import Counter
import pandas as pd


def prefixspan(db, min_sup, prefix=(), hasil=None):              #<1>
    hasil = {} if hasil is None else hasil
    dukung = Counter()
    for s in db:
        dukung.update(set(s))
    for item, n in sorted(dukung.items()):
        if n < min_sup:
            continue
        pola = prefix + (item,)
        hasil[pola] = n
        proyeksi = [s[s.index(item) + 1:] for s in db if item in s]
        prefixspan([p for p in proyeksi if p], min_sup, pola, hasil)
    return hasil


mini = ["B P M K", "B M D K", "P M D", "B M K"]                  #<2>
db = [s.split() for s in mini]
for pola, n in prefixspan(db, 3).items():
    print(" -> ".join(pola), f"(dukungan {n}/4)")

d = pd.read_csv("kode/data/sesi_aplikasi.csv")                   #<3>
seq = d.groupby("id_sesi").kejadian.apply(list)
beli = seq.map(lambda s: s[-1] == "selesai")
inti = seq.map(lambda s: [x for x in s[:-1] if x != "beranda"])
db = [tuple(dict.fromkeys(s)) for s in inti]     # buang ulangan

pola = prefixspan([list(s) for s in db], min_sup=250)            #<4>
print(f"\n{len(pola)} pola dengan dukungan >= 250 sesi (5%)")


def memuat(s, p):
    it = iter(s)
    return all(x in it for x in p)


baris = []                                                       #<5>
for p, n in pola.items():
    if len(p) >= 2 and "bayar" not in p:
        cocok = [memuat(s, p) for s in db]
        baris.append((" > ".join(p), n, beli[cocok].mean()))
tab = pd.DataFrame(baris, columns=["pola", "dukungan", "p_beli"])
print("rata-rata p_beli seluruh sesi:", round(beli.mean(), 3))
print(tab.sort_values("p_beli").iloc[[0, 1, 2, -3, -2, -1]]
      .to_string(index=False, float_format="%.2f"))
