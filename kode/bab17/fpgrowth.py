import time
from mlxtend.frequent_patterns import apriori, fpgrowth
from siapkan import keranjang, muat_struk

t = muat_struk()
B = keranjang(t)                                                 #<1>
print(f"{B.shape[0]} struk, {B.shape[1]} menu; rata-rata "
      f"{B.sum(1).mean():.2f} item per struk; kepadatan {B.values.mean():.3f}")
for minsup in [0.01, 0.003, 0.001]:
    for nama, f in [("Apriori", apriori), ("FP-Growth", fpgrowth)]:
        t0 = time.perf_counter()
        hasil = f(B, min_support=minsup, use_colnames=True)      #<2>
        print(f"minsup {minsup:5.3f} {nama:9s}: {len(hasil):4d} itemset, "
              f"{time.perf_counter() - t0:5.2f} dtk")
