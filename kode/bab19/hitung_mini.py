import sys; sys.path.insert(0, "kode/bab19")
from jb import JaringanBayes

Y, T = "ya", "tidak"
jb = JaringanBayes(                                              #<1>
    induk={"H": [], "P": [], "K": ["H", "P"], "J": ["K"], "C": ["J"]},
    nilai={v: [Y, T] for v in "HPKJC"})
def biner(p): return {Y: p, T: 1 - p}
jb.cpt = {                                                       #<2>
    "H": {(): biner(.4)},
    "P": {(): biner(.4)},
    "K": {(Y, Y): biner(.7), (Y, T): biner(.4),
          (T, Y): biner(.3), (T, T): biner(.1)},
    "J": {(Y,): biner(.5), (T,): biner(.05)},
    "C": {(Y,): biner(.8), (T,): biner(.1)},
}
def p(target, **bukti): return jb.tanya(target, bukti)[Y]       #<3>

print(f"P(C)               = {p('C'):.4f}")
print(f"P(C | H)           = {p('C', H=Y):.4f}   prediksi")
print(f"P(H | C)           = {p('H', C=Y):.4f}   diagnosis")
print("bukti berurutan:")                                        #<4>
print(f"  P(H)             = {p('H'):.4f}")
print(f"  P(H | C)         = {p('H', C=Y):.4f}")
print(f"  P(H | C,K)       = {p('H', C=Y, K=Y):.4f}")
print(f"  P(H | C,K,P)     = {p('H', C=Y, K=Y, P=Y):.4f}"
      "   explaining away")
print(f"  P(H | C,K,P,J)   = {p('H', C=Y, K=Y, P=Y, J=Y):.4f}"
      "   J tak menambah")
print(f"P(H | K)           = {p('H', K=Y):.4f}   = P(H | C,K)")
print(f"P(J | C)           = {p('J', C=Y):.4f}")
