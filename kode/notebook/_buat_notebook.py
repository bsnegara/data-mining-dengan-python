"""Membangun notebook pendamping dari naskah bab dan skrip kode/babNN.

Setiap notebook memuat sel persiapan (salin repositori, pasang pustaka),
lalu setiap listing bab dalam urutan kemunculannya di buku.
Pemakaian (dari akar repositori):  python kode/notebook/_buat_notebook.py
"""
import glob
import json
import re
from pathlib import Path

REPO = "https://github.com/bsnegara/data-mining-dengan-python"
POLA = re.compile(r"\\kodeberkas\{(.*?)\}\{[^}]*\}\{([^}]*)\}")


def sel(jenis, teks):
    baris = teks.rstrip("\n").split("\n")
    isi = [b + "\n" for b in baris[:-1]] + [baris[-1]]
    d = {"cell_type": jenis, "metadata": {}, "source": isi}
    if jenis == "code":
        d.update(execution_count=None, outputs=[])
    return d


def bersih_latex(t):
    t = re.sub(r"\\(emph|textit|textbf)\{(.*?)\}", r"\2", t)
    return t.replace("--", "–").replace("\\", "").replace("~", " ")


def persiapan(nn):
    return f"""# Jalankan sel ini sekali di awal sesi.
import os, sys, subprocess
if "google.colab" in sys.modules and not os.path.exists("kode"):
    if not os.path.exists("data-mining-dengan-python"):
        subprocess.run(["git", "clone", "--depth", "1",
                        "{REPO}"])
    os.chdir("data-mining-dengan-python")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "kode/requirements-colab.txt"])
while not os.path.exists("kode/data") and os.getcwd() != "/":
    os.chdir("..")                     # cari akar repositori
sys.path.insert(0, "kode/bab{nn}")
print("Folder kerja:", os.getcwd())"""


def main():
    for tex in sorted(glob.glob("bab/bab*.tex")):
        s = open(tex, encoding="utf-8").read()
        listing = POLA.findall(s)
        nb = re.search(r"\\notebook\{([^}]*)\}", s)
        if not listing or not nb:
            continue
        nn = Path(tex).stem[3:5]
        judul = bersih_latex(re.search(r"\\chapter\{(.*?)\}", s).group(1))
        sel_sel = [
            sel("markdown", f"# Bab {int(nn)}: {judul}\n\n"
                "Notebook pendamping buku *Data Mining dengan Python: "
                "Konsep, Algoritme, dan Studi Kasus dengan CRISP-DM*. "
                "Setiap sel kode sama dengan listing di buku; keluaran "
                "yang dicetak di buku ada di berkas `.out` di folder "
                f"`kode/bab{nn}/`.\n\nJalankan sel secara berurutan."),
            sel("code", persiapan(nn))]
        for i, (jdl, jalur) in enumerate(listing, 1):
            kode = open(jalur, encoding="utf-8").read()
            kode = re.sub(r"[ \t]*#<\d+>", "", kode).rstrip() + "\n"
            if "siapkan" in kode and "/aplikasi/" not in jalur:
                # modul siapkan ada di beberapa bab: pastikan versi bab ini
                kode = ("import sys; sys.modules.pop('siapkan', None); "
                        f"sys.path.insert(0, 'kode/bab{nn}')\n") + kode
            md = f"## Listing {int(nn)}.{i}: {bersih_latex(jdl)}\n\n`{jalur}`"
            if "/aplikasi/" in jalur:
                md += ("\n\nSel ini hanya menulis berkas aplikasi; "
                       "jalankan aplikasinya dari terminal seperti "
                       "dijelaskan di buku.")
                kode = f"%%writefile {jalur}\n" + kode
            sel_sel += [sel("markdown", md), sel("code", kode)]
        isi = {"cells": sel_sel, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}
        keluar = Path("kode/notebook") / f"{nb.group(1)}.ipynb"
        with open(keluar, "w", encoding="utf-8") as f:
            json.dump(isi, f, ensure_ascii=False, indent=1)
        print(f"{keluar}: {len(listing)} listing")


if __name__ == "__main__":
    main()
