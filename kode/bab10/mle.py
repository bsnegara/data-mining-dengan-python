import numpy as np
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from gerai_tahunan import matang

X = np.c_[np.ones(len(matang)), matang[["luas_m2", "kota_besar"]]]
y = matang["omzet"].to_numpy()


def nll(theta):                                                  #<1>
    b, log_s = theta[:-1], theta[-1]
    s2 = np.exp(2 * log_s)
    e = y - X @ b
    return 0.5 * len(y) * np.log(2 * np.pi * s2) + (e @ e) / (2 * s2)


hasil = minimize(nll, x0=np.r_[y.mean(), 0, 0, np.log(y.std())],
                 method="BFGS")                                  #<2>
ols = smf.ols("omzet ~ luas_m2 + kota_besar", matang).fit()
print("MLE :", hasil.x[:-1].round(3), " sigma", np.exp(hasil.x[-1]).round(2))
print("OLS :", ols.params.to_numpy().round(3),
      " sigma", np.sqrt(ols.ssr / len(y)).round(2))              #<3>
print(f"log-likelihood maksimum: {-hasil.fun:.2f} (statsmodels "
      f"{ols.llf:.2f})")
