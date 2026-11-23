import sys; sys.path.insert(0, "kode/bab19")
import numpy as np
from jb import muat_biji

d = muat_biji().drop(columns="tanggal")
V = list(d.columns)
N = len(d)

def skor_lokal(v, pa):                                           #<1>
    """BIC satu simpul: log-likelihood dikurangi penalti."""
    r = d[v].nunique()
    if pa:
        hit = d.groupby(list(pa) + [v]).size().unstack(fill_value=0)
        q = int(np.prod([d[p].nunique() for p in pa]))
    else:
        hit = d[v].value_counts().to_frame().T
        q = 1
    n = hit.to_numpy(float)
    nj = n.sum(1, keepdims=True)
    ll = (n * np.log(np.where(n > 0, n / nj, 1))).sum()
    return ll - 0.5 * np.log(N) * (r - 1) * q

def ada_siklus(induk):
    kunjung, tumpuk = set(), set()
    def dfs(v):
        if v in tumpuk: return True
        if v in kunjung: return False
        kunjung.add(v); tumpuk.add(v)
        hasil = any(dfs(p) for p in induk[v])
        tumpuk.discard(v)
        return hasil
    return any(dfs(v) for v in induk)

def calon_langkah(induk, a, b, boleh, maks):
    """Tambah a->b, atau hapus/balik a->b bila sudah ada."""
    if a in induk[b]:
        tanpa = tuple(p for p in induk[b] if p != a)
        yield {b: tanpa}
        if boleh(b, a) and len(induk[a]) < maks:
            yield {b: tanpa, a: induk[a] + (b,)}
    elif boleh(a, b) and len(induk[b]) < maks:
        yield {b: induk[b] + (a,)}

def mendaki_bukit(boleh=lambda a, b: True, maks=3):              #<2>
    induk = {v: () for v in V}
    skor = {v: skor_lokal(v, ()) for v in V}
    while True:
        terbaik, langkah = 1e-9, None
        for a in V:
            for b in V:
                if a == b: continue
                for ubah in calon_langkah(induk, a, b, boleh, maks):
                    if ada_siklus({**induk, **ubah}): continue
                    beda = sum(skor_lokal(v, pa) - skor[v]
                               for v, pa in ubah.items())
                    if beda > terbaik:
                        terbaik, langkah = beda, ubah
        if langkah is None:
            return induk, sum(skor.values())
        for v, pa in langkah.items():
            induk[v] = tuple(sorted(pa))
            skor[v] = skor_lokal(v, induk[v])

def cetak(induk, skor):
    print(f"  BIC = {skor:.1f}")
    for v in V:
        if induk[v]:
            print(f"  {' , '.join(induk[v]):>20s} -> {v}")

print("tanpa pengetahuan domain:")                               #<3>
cetak(*mendaki_bukit())
lapis = {"musim": 0, "jemur": 0, "basah": 0, "simpan": 0,        #<4>
         "air": 1, "jamur": 2, "cacat": 3}
print("dengan urutan waktu (lapis):")
cetak(*mendaki_bukit(boleh=lambda a, b: lapis[a] < lapis[b]))
