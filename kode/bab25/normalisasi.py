import re
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory)

KAMUS = {"bgt": "banget", "ga": "tidak", "gak": "tidak",          #<1>
         "nggak": "tidak", "tp": "tapi", "udah": "sudah",
         "aja": "saja", "hp": "ponsel", "app": "aplikasi",
         "appnya": "aplikasinya", "mantul": "mantap",
         "asem": "asam", "kayak": "seperti", "buat": "untuk",
         "nugas": "mengerjakan_tugas", "lemot": "lambat"}
GABUNG = {"force close": "force_close", "worth it": "worth_it",   #<2>
          "v60": "v_enam_puluh", "ac nya": "acnya"}
NEGASI = {"tidak", "tak", "bukan", "belum", "kurang", "jangan"}
JAGA = NEGASI | {"tapi", "masih", "lama", "sudah", "terlalu",
                 "sangat", "banget", "naik", "beda", "bisa"}      #<3>
STOP = set(StopWordRemoverFactory().get_stop_words()) - JAGA
_stem = StemmerFactory().create_stemmer()


@lru_cache(maxsize=None)
def stem(kata):
    return _stem.stem(kata) if "_" not in kata else kata


def bersihkan(teks, stopword=True, stemming=True, negasi=True):  #<4>
    t = teks.lower()
    for a, b in GABUNG.items():
        t = t.replace(a, b)
    token = re.findall(r"[a-z_]+", t)
    token = [KAMUS.get(w, w) for w in token]
    if stopword:
        token = [w for w in token if w not in STOP]
    if stemming:
        token = [stem(w) for w in token]
    if negasi:                       # tidak enak -> tidak_enak
        hasil, i = [], 0
        while i < len(token):
            if token[i] in NEGASI and i + 1 < len(token):
                hasil.append(token[i] + "_" + token[i + 1]); i += 2
            else:
                hasil.append(token[i]); i += 1
        token = hasil
    return " ".join(token)
