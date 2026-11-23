import numpy as np

luas = np.array([75, 102, 59, 74, 40, 110])                      #<1>
omzet = np.array([1780, 1780, 1640, 1630, 1470, 1700])           # ribu Rp
xb, yb = luas.mean(), omzet.mean()
sxx = ((luas - xb) ** 2).sum()                                   #<2>
sxy = ((luas - xb) * (omzet - yb)).sum()
b1 = sxy / sxx
b0 = yb - b1 * xb
yhat = b0 + b1 * luas
jkg = ((omzet - yhat) ** 2).sum()                                #<3>
jkt = ((omzet - yb) ** 2).sum()
print(f"x rata-rata {xb:.3f}, y rata-rata {yb:.3f}")
print(f"Sxx = {sxx:.2f}, Sxy = {sxy:.2f}")
print(f"b1 = {b1:.4f} ribu/m2, b0 = {b0:.2f} ribu")
print(f"JKG = {jkg:.1f}, JKT = {jkt:.1f}, R2 = {1 - jkg / jkt:.3f}")
print("sisa:", np.round(omzet - yhat, 1))
print("cek polyfit:", np.round(np.polyfit(luas, omzet, 1), 4))  #<4>
