import numpy as np
from siapkan import KEJADIAN, SELESAI, muat, pad
from hmm_beralih import maju

seq, info, _ = muat()
lat = seq[info.latih]
n = np.zeros((9, 9))
for s in lat:
    np.add.at(n, (s[:-1], s[1:]), 1)
P = n[:7] / n[:7].sum(1, keepdims=True)
mod = np.load("kode/bab24/_model.npz")
nama = ["beranda", "promo", "cari", "lihat menu", "detail menu",
        "keranjang", "bayar", "selesai", "keluar"]
w, h = 9.5, 5.6
out = [r"\begin{tikzpicture}[dl,font=\sffamily\scriptsize]"]
for i in range(7):
    out.append(rf"\node[anchor=east] at (-1mm,{-i*h-h/2:.1f}mm) {{{nama[i]}}};")
    for j in range(9):
        p = P[i, j]
        warna = "white" if p > .45 else "black"
        out.append(rf"\fill[dlBiru!{min(100, p*160):.0f}!white] "
                   rf"({j*w:.1f}mm,{-i*h:.1f}mm) rectangle ++({w:.1f}mm,{-h:.1f}mm);")
        if p >= .005:
            out.append(rf"\node[text={warna}] at ({j*w+w/2:.1f}mm,{-i*h-h/2:.1f}mm) {{{p:.2f}}};")
    b = mod["b_markov"][i]
    out.append(rf"\fill[dlOranye!{b*100:.0f}!white] ({9*w+3:.1f}mm,{-i*h:.1f}mm) rectangle ++({w+2:.1f}mm,{-h:.1f}mm);")
    out.append(rf"\node at ({9*w+3+(w+2)/2:.1f}mm,{-i*h-h/2:.1f}mm) {{{b:.2f}}};")
for j in range(9):
    out.append(rf"\node[anchor=west,rotate=50] at ({j*w+w/2-1:.1f}mm,1mm) {{{nama[j]}}};")
out.append(rf"\node[anchor=west,rotate=50] at ({9*w+3+(w+2)/2-1:.1f}mm,1mm) {{\textbf{{P(beli)}}}};")
out.append(rf"\node[keterangan,anchor=north] at ({4.5*w:.1f}mm,{-7*h-1:.1f}mm) {{kejadian berikutnya}};")
out.append(r"\end{tikzpicture}")
txt = "\n".join(out)
import re
txt = re.sub(r"\{(\d)\.(\d\d)\}", r"{\1,\2}", txt)
open("gambar/bab24/transisi.tex", "w").write(txt + "\n")

uji = seq[~info.latih]
Ou, Mu = pad(list(uji))
a, _ = maju(Ou, Mu, mod["pi"], mod["A"], mod["E"])
pn = a[..., int(mod["nt"])]
panel = []
for no, (i, judul) in enumerate([(2, "sesi berakhir membeli"),
                                 (28, "keranjang ditinggalkan")]):
    s = uji.iloc[i]
    lab = ",".join(nama[k].replace(" ", "~") for k in s)
    xy = " ".join(f"({t+1},{pn[i, t]:.3f})" for t in range(len(s) - 1))
    opsi = (f"name=p{no}" if no == 0 else
            "name=p1,at={(p0.east)},anchor=west,xshift=14mm")
    panel.append(rf"""\begin{{axis}}[{opsi},width=62mm,height=46mm,
  ymin=0,ymax=1,xmin=0.5,xmax={len(s)+.5},axis lines=left,
  xtick={{1,...,{len(s)}}},xticklabels={{{lab}}},
  x tick label style={{rotate=55,anchor=east,font=\sffamily\tiny}},
  y tick label style={{font=\sffamily\scriptsize,/pgf/number format/use comma}},
  ylabel={{$P(\text{{niat}}\mid o_{{1:t}})$}},label style={{font=\sffamily\scriptsize}},
  title={{{judul}}},title style={{font=\sffamily\small}}]
\addplot[dlAbu,dashed,domain=0.5:{len(s)+.5}] {{0.5}};
\addplot[dlHijau,very thick,mark=*,mark size=1.6pt] coordinates {{{xy}}};
\end{{axis}}""")
open("gambar/bab24/posterior.tex", "w").write(
    "\\begin{tikzpicture}\n" + "\n".join(panel) + "\n\\end{tikzpicture}\n")
