"""Menyusun gambar/bab20/biplot.tex dari skor.dat dan loading.dat."""
import re

def baca(f):
    baris = open(f).read().splitlines()[1:]
    return [re.match(r"(\S+) (\S+) (?:(\d) )?\{(.*)\}", b).groups() for b in baris]

skor = baca("gambar/bab20/skor.dat")
muat = baca("gambar/bab20/loading.dat")
POS = {"Kintamani": "above", "Lampung Robusta": "below right", "Aceh Robusta": "below left",
       "Dampit": "left", "Sidikalang": "above", "Gayo": "right", "Bajawa": "right",
       "Temanggung": "right", "Malino": "below", "Halmahera": "below left",
       "Pegunungan Bintang": "below", "Wamena": "above", "Preanger": "below left",
       "Solok": "below right", "Malabar": "below", "Ijen": "above",
       "Kerinci": "above", "Toraja": "above", "Enrekang": "right",
       "Manggarai": "below left", "Kalosi": "below", "Semendo": "left",
       "Mandailing": "above right", "Lintong": "below"}
o = [r"\begin{tikzpicture}",
     r"\begin{axis}[width=120mm,height=84mm,axis lines=middle,xmin=-7,xmax=8.6,ymin=-3.2,ymax=4.6,",
     r"  axis equal image=false,xtick={-6,-4,...,6},ytick={-2,0,2,4},",
     r"  tick label style={font=\sffamily\tiny},/pgf/number format/use comma,",
     r"  legend style={draw=none,font=\sffamily\scriptsize,at={(0.01,0.01)},anchor=south west},legend cell align=left]"]
for k, (gaya, nama) in enumerate([("mark=*,dlBiru,mark size=1.6pt", "arabika"),
                                  ("mark=square*,dmKopi,mark size=1.8pt", "robusta")]):
    pts = " ".join(f"({x},{y})" for x, y, r, _ in skor if int(r) == k)
    o += [rf"\addplot[only marks,{gaya}] coordinates {{{pts}}};", rf"\addlegendentry{{{nama}}}"]
for x, y, r, n in skor:
    o.append(rf"\node[font=\sffamily\tiny,text=black!70,{POS.get(n, 'right')},inner sep=1.5pt] at (axis cs:{x},{y}) {{{n}}};")
S = 4
for x, y, _, n in muat:
    o.append(rf"\draw[->,dlOranye!85!black,thin] (axis cs:0,0) -- (axis cs:{S*float(x):.2f},{S*float(y):.2f});")
o += [r"\node[font=\sffamily\scriptsize,text=dlOranye!80!black,align=left,anchor=west] at (axis cs:4.0,2.4) {tanah, pahit, rempah,\\kekentalan, cokelat};",
      r"\node[font=\sffamily\scriptsize,text=dlOranye!80!black,align=right,anchor=east] at (axis cs:-3.2,2.3) {buah, bunga, sitrus,\\keasaman, manis, aroma};",
      r"\node[font=\sffamily\scriptsize,text=dlOranye!80!black,anchor=south] at (axis cs:0,3.95) {rasa ikutan};",
      r"\node[font=\sffamily\scriptsize,anchor=south east] at (axis cs:8.6,0.15) {PC1: cerah $\leftrightarrow$ berat};",
      r"\end{axis}", r"\end{tikzpicture}"]
open("gambar/bab20/biplot.tex", "w").write("\n".join(o) + "\n")
