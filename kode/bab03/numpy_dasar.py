import numpy as np

harga = np.array([22000, 28000, 20000, 18000, 32000])  #<1>
qty = np.array([3, 1, 2, 4, 1])

omzet = harga * qty                                       #<2>
print("omzet per menu :", omzet)
print("total omzet    :", omzet.sum())

diskon = np.array([0.0, 0.1, 0.2])                        #<3>
harga_promo = harga[:, None] * (1 - diskon[None, :])
print("bentuk tabel   :", harga_promo.shape)
print(harga_promo[:2])

mahal = harga >= 25000                                    #<4>
print("menu mahal     :", harga[mahal])
print("rata-rata, simpangan baku:",
      harga.mean(), round(harga.std(), 1))
