"""Modul bantu Bab 17: keranjang struk dan item konteks."""
import pandas as pd

DATA = "kode/data/"


def muat_struk():
    t = pd.read_csv(DATA + "transaksi_lengkap_2026q1.csv",
                    usecols=["no_struk", "nama_menu", "kategori", "waktu"])
    t["waktu"] = pd.to_datetime(t["waktu"])
    return t


def keranjang(t, konteks=False, minimal=1):
    """Matriks boolean struk x item; opsional item konteks [pagi]/[berbuka]."""
    item = t[["no_struk", "nama_menu"]].rename(columns={"nama_menu": "item"})
    if konteks:
        w = t.groupby("no_struk")["waktu"].first()
        ramadan = (w >= "2026-02-18") & (w <= "2026-03-19")
        tambah = [pd.DataFrame({"no_struk": w.index[w.dt.hour < 10],
                                "item": "[pagi]"}),
                  pd.DataFrame({"no_struk": w.index[ramadan &
                                                    (w.dt.hour >= 17)],
                                "item": "[berbuka]"})]
        item = pd.concat([item] + tambah)
    n = t.groupby("no_struk").size()
    item = item[item["no_struk"].isin(n.index[n >= minimal])]
    return pd.crosstab(item["no_struk"], item["item"]).gt(0)
