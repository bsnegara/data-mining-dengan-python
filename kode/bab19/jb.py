"""Jaringan Bayes diskret kecil: CPT, inferensi enumerasi, intervensi.

Cukup untuk belasan variabel diskret; untuk jaringan besar gunakan
pustaka seperti pgmpy.
"""
import itertools

import numpy as np
import pandas as pd


def muat_biji(path="kode/data/mutu_biji.csv"):                  #<1>
    """Diskretkan data mutu biji menjadi tujuh variabel kategorik."""
    d = pd.read_csv(path, parse_dates=["tanggal"])
    ya = lambda b, s1, s0: np.where(b, s1, s0)
    x = pd.DataFrame({
        "musim": d.musim_panen,
        "jemur": d.cara_jemur,
        "basah": ya(d.proses == "giling basah", "ya", "tidak"),
        "simpan": ya(d.lama_simpan_hari > 60, "lama", "singkat"),
        "air": ya(d.kadar_air > 12.5, "tinggi", "normal"),
        "jamur": ya(d.jamur == 1, "ya", "tidak"),
        "cacat": ya(d.cacat_primer > 5, "ya", "tidak"),
    })
    x["tanggal"] = d.tanggal
    return x


class JaringanBayes:
    def __init__(self, induk, nilai):
        self.induk = induk            # {simpul: [induk, ...]}
        self.nilai = nilai            # {simpul: [nilai yang mungkin]}
        self.cpt = {}

    def latih(self, data, alfa=1.0):                             #<2>
        """Taksir CPT: hitungan frekuensi + pemulusan Laplace."""
        for v, pa in self.induk.items():
            nilai_pa = [self.nilai[p] for p in pa]
            tabel = {}
            for k in itertools.product(*nilai_pa):
                bagian = data
                for p, nilai_p in zip(pa, k):
                    bagian = bagian[bagian[p] == nilai_p]
                hit = bagian[v].value_counts()
                n = np.array([hit.get(s, 0) for s in self.nilai[v]])
                n = n + alfa
                tabel[k] = dict(zip(self.nilai[v], n / n.sum()))
            self.cpt[v] = tabel
        return self

    def gabungan(self, intervensi=None):
        """Distribusi gabungan semua variabel sebagai DataFrame.

        intervensi={v: nilai} memotong panah yang masuk ke v dan
        menetapkan nilainya (operator do).
        """
        intervensi = intervensi or {}
        simpul = list(self.induk)
        baris = []
        semua = [self.nilai[v] for v in simpul]
        for konf in itertools.product(*semua):
            a = dict(zip(simpul, konf))
            p = 1.0
            for v in simpul:
                if v in intervensi:                              #<3>
                    p *= 1.0 if a[v] == intervensi[v] else 0.0
                else:
                    k = tuple(a[q] for q in self.induk[v])
                    p *= self.cpt[v][k][a[v]]
                if p == 0:
                    break
            a["p"] = p
            baris.append(a)
        return pd.DataFrame(baris)

    def tanya(self, target, bukti=None, intervensi=None):        #<4>
        """P(target | bukti) atau P(target | do(iv), bukti)."""
        g = self.gabungan(intervensi)
        for v, s in (bukti or {}).items():
            g = g[g[v] == s]
        return g.groupby(target).p.sum() / g.p.sum()

    def log_lik(self, data):
        ll = 0.0
        for v, pa in self.induk.items():
            for _, r in data.iterrows():
                k = tuple(r[q] for q in pa)
                ll += np.log(self.cpt[v][k][r[v]])
        return ll
