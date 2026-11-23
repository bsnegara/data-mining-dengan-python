import statsmodels.formula.api as smf
from gerai_tahunan import matang

print(f"{len(matang)} gerai, omzet harian rata-rata 2025 (ribu Rp)")
m1 = smf.ols("omzet ~ jumlah_kursi", matang).fit()               #<1>
print(f"\nsederhana: kursi {m1.params['jumlah_kursi']:+.2f} "
      f"(p = {m1.pvalues['jumlah_kursi']:.2f}), R2 = {m1.rsquared:.3f}")
m2 = smf.ols("omzet ~ jumlah_kursi + luas_m2 + C(tipe_lokasi)"   #<2>
             " + kota_besar", matang).fit()
tab = m2.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]]
tab.index = [i.replace("C(tipe_lokasi)[T.", "tipe ").replace("]", "")
             for i in tab.index]
print("\nberganda:")
print(tab.round(3).to_string())
print(f"R2 = {m2.rsquared:.3f}, R2 terkoreksi = {m2.rsquared_adj:.3f}, "
      f"sigma = {m2.mse_resid ** 0.5:.1f}")                      #<3>
print(f"uji F: F = {m2.fvalue:.1f}, p = {m2.f_pvalue:.1e}")
