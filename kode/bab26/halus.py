import numpy as np
import pandas as pd
import pywt


def halus_wavelet(x, wavelet="sym8", tingkat=5, mode="soft"):
    """Penghilangan derau: ambang universal pada koefisien rincian."""
    c = pywt.wavedec(x, wavelet, level=tingkat, mode="smooth")
    sigma = np.median(np.abs(c[-1])) / 0.6745          # MAD tingkat 1
    ambang = sigma * np.sqrt(2 * np.log(len(x)))
    c = [c[0]] + [pywt.threshold(d, ambang, mode) for d in c[1:]]
    return pywt.waverec(c, wavelet, mode="smooth")[:len(x)]


def halus_wavelet_ti(x, geser=16, **opsi):
    """Invarian geser (cycle spinning): rata-rata hasil dari sinyal
    yang dipotong k titik di depan, k = 0, ..., geser-1."""
    jumlah, n = np.zeros(len(x)), np.zeros(len(x))
    for k in range(geser):
        jumlah[k:] += halus_wavelet(x[k:], **opsi)
        n[k:] += 1
    return jumlah / n


def halus_median(x, w=9):
    return pd.Series(x).rolling(w, center=True,
                                min_periods=1).median().to_numpy()


def halus_rata(x, w=15):
    return pd.Series(x).rolling(w, center=True,
                                min_periods=1).mean().to_numpy()


def ror(x, w=30):
    """Laju kenaikan suhu (derajat per menit) dari selisih w detik."""
    return pd.Series(x).diff(w).to_numpy() * 60 / w
