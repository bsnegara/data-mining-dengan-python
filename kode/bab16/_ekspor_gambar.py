"""Membuat berkas TikZ dendrogram dan peta SOM untuk Bab 16."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

WARNA = {"kampus": "dlHijau", "mal": "dlBiru", "perkantoran": "dlOranye",
         "permukiman": "dlUngu"}


def tikz_dendro(L, label, warna_daun, lebar, tinggi, skala_y, ylab,
                putar=True):
    d = dendrogram(L, no_plot=True, labels=label)
    sx = lebar / (10 * len(label))
    out = []
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        pts = " -- ".join(f"({x * sx:.2f},{y * skala_y:.2f})"
                          for x, y in zip(xs, ys))
        out.append(f"  \\draw[dlAbu!80!black,thick] {pts};")
    for i, n in enumerate(d["ivl"]):
        x = (10 * i + 5) * sx
        gaya = ("font=\\sffamily\\tiny,rotate=90,anchor=east" if putar
                else "font=\\sffamily\\scriptsize,anchor=north")
        out.append(f"  \\node[{gaya},text={warna_daun(n)}] at "
                   f"({x:.2f},-0.6) {{{n}}};")
    ymax = max(max(y) for y in d["dcoord"])
    langkah = 0.1 if ymax < 1 else (2 if ymax > 5 else 0.5)
    fmt = 1
    t = 0.0
    out.append(f"  \\draw[dlAbu,->] (-1,0) -- (-1,{ymax * skala_y + 2:.2f});")
    while t <= ymax + 1e-9:
        y = t * skala_y
        teks = f"{t:.1f}".replace(".", "{,}")
        out.append(f"  \\draw[dlAbu] (-1.6,{y:.2f}) -- (-1,{y:.2f}) "
                   f"node[left=3pt,font=\\sffamily\\tiny] {{{teks}}};")
        t += langkah
    out.append(f"  \\node[font=\\sffamily\\scriptsize,rotate=90] at "
               f"(-10,{ymax * skala_y / 2:.2f}) {{{ylab}}};")
    return out


X = np.array([[0.85, 0.80], [0.90, 0.85], [0.85, 1.00],
              [1.15, 1.00], [1.10, 1.05], [1.20, 0.95]])
nama = [f"G{i}" for i in range(1, 7)]
isi = ["\\begin{tikzpicture}[x=1mm,y=1mm]"]
for k, m in enumerate(["single", "complete"]):
    isi.append(f"\\begin{{scope}}[xshift={k * 62}mm]")
    isi.append(f"  \\node[font=\\sffamily\\small\\bfseries,text=dlBiru] at "
               f"(22,44) {{({'ab'[k]}) {m} linkage}};")
    isi += tikz_dendro(linkage(X, m), nama, lambda n: "black", 45, 40,
                       100, "jarak", putar=False)
    isi.append("\\end{scope}")
isi.append("\\end{tikzpicture}")
open("gambar/bab16/dendro-mini.tex", "w").write("\n".join(isi) + "\n")

L = np.load("kode/bab16/linkage_gerai.npy")
kl = pd.read_csv("kode/bab16/klaster_gerai.csv", index_col=0)
gerai = pd.read_csv("kode/data/gerai.csv").set_index("id_gerai")
lab = list(kl.index)
isi = ["\\begin{tikzpicture}[x=1mm,y=1mm]"]
isi += tikz_dendro(L, lab, lambda n: WARNA[gerai.loc[n, "tipe_lokasi"]],
                   110, 50, 5, "jarak Ward")
isi.append("  \\draw[dlMerah,dashed] (0,{:.2f}) -- (110,{:.2f}) "
           "node[right,font=\\sffamily\\tiny,text=dlMerah] {{$k=4$}};"
           .format(3.9 * 5, 3.9 * 5))
for i, (t, w) in enumerate(WARNA.items()):
    isi.append(f"  \\node[font=\\sffamily\\scriptsize,text={w},anchor=west]"
               f" at ({4 + 26 * i},-11) {{$\\blacksquare$ {t}}};")
isi.append("\\end{tikzpicture}")
open("gambar/bab16/dendro-gerai.tex", "w").write("\n".join(isi) + "\n")

s = pd.read_csv("gambar/bab16/som.dat", sep=" ")
WS = {0: "dlOranye", 1: "dlBiru", 2: "dlHijau", 3: "dlMerah", 4: "dlUngu",
      -1: "white"}
nmax = s["n"].max()
isi = ["\\begin{tikzpicture}[x=6mm,y=6mm]"]
for _, r in s.iterrows():
    x, y = r["i"], 7 - r["j"]
    isi.append(f"  \\fill[{WS[int(r['seg'])]}!{int(35 + 60 * r['u'] / s['u'].max())}]"
               f" ({x},{y}) rectangle +(1,1);")
    if r["n"] > 0:
        rad = 0.05 + 0.25 * np.sqrt(r["n"] / nmax)
        isi.append(f"  \\fill[white] ({x + 0.5},{y + 0.5}) circle ({rad:.2f});")
isi.append("  \\draw[dlAbu!60] (0,0) grid (8,8);")
for i, (k, t) in enumerate([(0, "mahasiswa"), (1, "pekerja pagi"),
                            (2, "keluarga"), (3, "tidur"), (4, "penikmat")]):
    isi.append(f"  \\node[font=\\sffamily\\scriptsize,anchor=west] at "
               f"(8.4,{7.3 - 0.8 * i}) {{\\textcolor{{{WS[k]}}}"
               f"{{$\\blacksquare$}} {t}}};")
isi.append("  \\node[font=\\sffamily\\scriptsize,anchor=west,align=left] at "
           "(8.4,2.4) {lingkaran putih:\\\\banyak anggota;\\\\warna pekat:\\\\"
           "jarak U besar};")
isi.append("\\end{tikzpicture}")
open("gambar/bab16/som.tex", "w").write("\n".join(isi) + "\n")
