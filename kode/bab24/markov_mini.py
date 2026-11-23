import numpy as np
import pandas as pd

sesi = ["M K S", "M M K S", "M K M K S",                        #<1>
        "M M X", "M M K X", "M X"]
keadaan = ["M", "K", "S", "X"]
hitung = pd.DataFrame(0, index=keadaan, columns=keadaan)
for s in sesi:
    x = s.split()
    for a, b in zip(x[:-1], x[1:]):
        hitung.loc[a, b] += 1
print("banyak transisi:\n", hitung.loc[["M", "K"]])

P = hitung.loc[["M", "K"]].div(hitung.loc[["M", "K"]].sum(1),   #<2>
                                axis=0)
print("\npeluang transisi:\n", P.round(2))

Q = P[["M", "K"]].to_numpy()                                     #<3>
R = P[["S", "X"]].to_numpy()
N = np.linalg.inv(np.eye(2) - Q)
print("\nmatriks fundamental N:\n", N.round(3))
print("harapan langkah sampai selesai:", N.sum(1).round(2))
B = N @ R
print("peluang berakhir di S / X:")
for i, k in enumerate(["M", "K"]):
    print(f"  dari {k}: S = {B[i, 0]:.2f}, X = {B[i, 1]:.2f}")

P_penuh = np.zeros((4, 4))                                       #<4>
P_penuh[:2] = P.to_numpy()
P_penuh[2, 2] = P_penuh[3, 3] = 1
P2 = np.linalg.matrix_power(P_penuh, 2)
print("\nP(M ke S dalam tepat 2 langkah) =", round(P2[0, 2], 2))
P50 = np.linalg.matrix_power(P_penuh, 50)
print("P^50 baris M:", P50[0].round(3))
