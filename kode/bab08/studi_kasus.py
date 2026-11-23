import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from kartu import df, kartu, y                                   #<1>

cv = StratifiedKFold(10, shuffle=True, random_state=0)
semua = ["umur", "frekuensi_90h", "frekuensi_90h_lalu", "recency_hari",
         "nilai_90h", "rata_nilai_struk", "porsi_promo", "porsi_aplikasi",
         "porsi_kopi_hitam", "porsi_pagi", "porsi_akhir_pekan",
         "ragam_menu", "keluhan_90h", "jarak_km", "pakai_ulasan"]
logistik = make_pipeline(ColumnTransformer([
    ("n", StandardScaler(), semua),
    ("k", OneHotEncoder(), ["kota", "tingkat", "jenis_kelamin"])]),
    LogisticRegression(max_iter=2000))
calon = {"Gaussian NB, 15 atribut": (GaussianNB(), df[semua]),   #<2>
         "NB kartu, 7 atribut": (CategoricalNB(alpha=1), kartu),
         "logistik, 18 atribut": (logistik, df)}
for nama, (m, X) in calon.items():
    s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
    print(f"{nama:24s} AUC {s.mean():.3f} +/- {s.std():.3f}")

nb = CategoricalNB(alpha=1).fit(kartu, y)
print("\nKartu skor (poin = 10 x log10 P(x|churn)/P(x|setia)):")
for j, atr in enumerate(kartu.columns):                          #<3>
    lp = nb.feature_log_prob_[j]
    poin = np.round(10 * (lp[1] - lp[0]) / np.log(10)).astype(int)
    print(f"  {atr:13s}", " ".join(f"{p:+d}" for p in poin))
awal = 10 * np.log10(y.mean() / (1 - y.mean()))                  #<4>
print(f"poin awal (peluang prior): {awal:.0f}")

poin_total = np.full(len(kartu), awal)                           #<5>
for j, atr in enumerate(kartu.columns):
    lp = nb.feature_log_prob_[j]
    poin = np.round(10 * (lp[1] - lp[0]) / np.log(10))
    poin_total += poin[kartu[atr].to_numpy()]
kelompok = np.digitize(poin_total, [-30, -20, -10, 0, 10])
nama = ["< -30", "-30..-21", "-20..-11", "-10..-1", "0..9", ">= 10"]
print("\npoin total  anggota  churn nyata  peluang NB")
for g in range(6):
    m = kelompok == g
    p_nb = 1 / (1 + 10 ** (-poin_total[m] / 10))
    print(f"{nama[g]:>9s}   {m.sum():6d}      {y[m].mean():.3f}"
          f"       {p_nb.mean():.3f}")
