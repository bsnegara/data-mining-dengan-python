from itertools import combinations

struk = [{"gula aren", "roti", "pisang"}, {"gula aren", "roti"},     #<1>
         {"americano", "croissant"}, {"gula aren", "pisang"},
         {"gula aren", "roti", "teh"}, {"americano", "croissant", "roti"},
         {"gula aren", "roti", "pisang"}, {"teh", "pisang"},
         {"gula aren", "teh"}]
n, minsup = len(struk), 0.2
sup = lambda s: sum(s <= t for t in struk) / n                   #<2>

item = sorted(set().union(*struk))
L = [frozenset([i]) for i in item if sup({i}) >= minsup]
semua = {s: sup(s) for s in L}
k = 1
print(f"L1: " + ", ".join(f"{set(s)} {sup(s):.2f}" for s in L))
while L:
    k += 1
    C = {a | b for a in L for b in L if len(a | b) == k}         #<3>
    C = {c for c in C if all(frozenset(s) in semua                #<4>
                             for s in combinations(c, k - 1))}
    L = [c for c in C if sup(c) >= minsup]
    if C:
        print(f"C{k}: {len(C)} kandidat; L{k}: " + ", ".join(
            f"{sorted(s)} {sup(s):.2f}" for s in sorted(L, key=sorted)))
    semua.update({s: sup(s) for s in L})

print("\naturan dengan confidence >= 0.6:")
for s in [s for s in semua if len(s) >= 2]:                      #<5>
    for r in range(1, len(s)):
        for a in map(frozenset, combinations(s, r)):
            conf = semua[s] / semua[a]
            lift = conf / semua[s - a]
            if conf >= 0.6:
                print(f"  {sorted(a)} -> {sorted(s - a)}: sup {semua[s]:.2f}"
                      f" conf {conf:.2f} lift {lift:.2f}")
