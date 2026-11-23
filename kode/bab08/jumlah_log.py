import numpy as np

p = np.full(400, 0.3)                                            #<1>
print("hasil kali langsung :", np.prod(p))
print("jumlah logaritma    :", np.log(p).sum().round(2))
la, lb = -480.2, -483.9                                          #<2>
m = max(la, lb)
pa = np.exp(la - m) / (np.exp(la - m) + np.exp(lb - m))
print(f"P(A | x) dari log   : {pa:.4f}")
