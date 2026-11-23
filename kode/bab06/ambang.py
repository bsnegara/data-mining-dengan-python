import numpy as np

kadar = np.array([11.2, 11.8, 12.1, 12.4, 12.6, 13.0, 13.5, 14.1])
label = np.array([0, 0, 0, 1, 0, 1, 1, 1])        # 1 = tolak

def gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 1 - p ** 2 - (1 - p) ** 2

calon = (kadar[:-1] + kadar[1:]) / 2                             #<1>
for t in calon:
    kiri, kanan = label[kadar <= t], label[kadar > t]
    g = (len(kiri) * gini(kiri) + len(kanan) * gini(kanan)) / len(label)
    print(f"ambang {t:5.2f}: Gini terbobot = {g:.3f}")           #<2>
