"""Menyusun gambar Bab 21 yang memerlukan label titik."""
import re

def baca(f, kepala=True):
    b = open(f).read().splitlines()[1 if kepala else 0:]
    return [re.match(r"(\S+) (\S+) (?:(\d) )?\{(.*)\}", x).groups() for x in b]

# peta kota
kota = baca("gambar/bab21/kota.dat")
POSK = {"Padang": "below", "Bengkulu": "below", "Jambi": "above", "Palembang": "below",
        "Pekanbaru": "above", "Medan": "above", "Banda Aceh": "above", "Bandar Lampung": "above left"}
o = [r"\begin{tikzpicture}",
     r"\begin{axis}[width=100mm,height=50mm,axis lines=left,xmin=-1100,xmax=850,ymin=-260,ymax=220,",
     r"  xlabel={dimensi 1 (km)},ylabel={dimensi 2 (km)},",
     r"  tick label style={font=\sffamily\scriptsize},label style={font=\sffamily\scriptsize},",
     r"  /pgf/number format/1000 sep={.},axis equal image]",
     r"\addplot[only marks,mark=*,dlBiru,mark size=1.8pt] coordinates {" +
     " ".join(f"({x},{y})" for x, y, _, _ in kota) + "};"]
for x, y, _, n in kota:
    o.append(rf"\node[font=\sffamily\scriptsize,{POSK[n]},inner sep=2pt] at (axis cs:{x},{y}) {{{n}}};")
o += [r"\end{axis}", r"\end{tikzpicture}"]
open("gambar/bab21/kota.tex", "w").write("\n".join(o) + "\n")

# peta rasa
peta = baca("gambar/bab21/peta.dat")
panah = baca("gambar/bab21/panah.dat", kepala=False)
GAYA = ["mark=*,dlBiru", "mark=square*,dlOranye", "mark=triangle*,dlHijau,mark size=2.3pt",
        "mark=diamond*,dlUngu,mark size=2.3pt", "mark=pentagon*,dmKopi"]
PULAU = ["Sumatera", "Jawa", "Bali--Nusa Tenggara", "Sulawesi", "Papua--Maluku"]
POS = {"Lampung Robusta": "below", "Aceh Robusta": "left", "Dampit": "right",
       "Kintamani": "right", "Bajawa": "above", "Gayo": "above", "Mandailing": "right",
       "Sidikalang": "below", "Lintong": "right", "Halmahera": "below", "Manggarai": "left",
       "Wamena": "left", "Pegunungan Bintang": "above left", "Ijen": "above right", "Malino": "below",
       "Toraja": "above", "Kalosi": "right", "Enrekang": "below", "Semendo": "below",
       "Temanggung": "right", "Malabar": "below", "Solok": "right", "Preanger": "left",
       "Kerinci": "left"}
o = [r"\begin{tikzpicture}",
     r"\begin{axis}[width=104mm,height=92mm,axis lines=middle,xmin=-2.3,xmax=2.3,ymin=-2.4,ymax=2.3,",
     r"  xtick=\empty,ytick=\empty,clip=false,",
     r"  legend style={draw=none,font=\sffamily\scriptsize,at={(1.1,0.5)},anchor=west},legend cell align=left]"]
for k, g in enumerate(GAYA):
    pts = " ".join(f"({x},{y})" for x, y, p, _ in peta if int(p) == k)
    o += [rf"\addplot[only marks,{g},mark size=1.8pt] coordinates {{{pts}}};",
          rf"\addlegendentry{{{PULAU[k]}}}"]
for x, y, _, n in peta:
    o.append(rf"\node[font=\sffamily\tiny,text=black!75,{POS.get(n, 'right')},inner sep=1.5pt] at (axis cs:{x},{y}) {{{n}}};")
S = 1.9
GABUNG = {"tanah": "tanah, pahit", "bunga": "bunga, keasaman",
          "pahit": "", "keasaman": ""}
for x, y, _, n in panah:
    n = GABUNG.get(n, n)
    tx, ty = float(x) * S, float(y) * S
    o.append(rf"\draw[->,dlMerah!70!black,thin] (axis cs:0,0) -- (axis cs:{tx:.2f},{ty:.2f}) "
             rf"node[font=\sffamily\tiny\itshape,text=dlMerah!70!black,anchor={'west' if tx >= 0 else 'east'}] {{{n}}};")
o += [r"\node[font=\sffamily\scriptsize,anchor=south east] at (axis cs:2.3,0.03) {berat};",
      r"\node[font=\sffamily\scriptsize,anchor=south west] at (axis cs:-2.3,0.03) {cerah};",
      r"\node[font=\sffamily\scriptsize,anchor=north west] at (axis cs:0.03,2.3) {manis, bulat};",
      r"\end{axis}", r"\end{tikzpicture}"]
open("gambar/bab21/peta.tex", "w").write("\n".join(o) + "\n")
