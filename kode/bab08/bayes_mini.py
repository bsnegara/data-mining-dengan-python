import pandas as pd

mini = pd.DataFrame({
    "tingkat":   ["perunggu", "perunggu", "perak", "perunggu", "perak",
                  "perunggu", "perak", "perak", "perunggu", "perak"],
    "kunjungan": ["jarang", "jarang", "jarang", "sering", "sering",
                  "sering", "sering", "jarang", "sering", "sering"],
    "keluhan":   ["ya", "tidak", "ya", "ya", "tidak",
                  "tidak", "tidak", "tidak", "ya", "tidak"],
    "promo":     ["tinggi", "tinggi", "tinggi", "tinggi", "rendah",
                  "tinggi", "rendah", "rendah", "rendah", "tinggi"],
    "churn":     ["ya", "ya", "ya", "ya", "tidak",
                  "tidak", "tidak", "tidak", "tidak", "tidak"]})


def naive_bayes(data, x, alfa=0):
    prior = data["churn"].value_counts(normalize=True)           #<1>
    skor = {}
    for c in prior.index:
        bagian = data[data["churn"] == c]
        p = prior[c]
        for atr, nilai in x.items():
            k = data[atr].nunique()                              #<2>
            p *= ((bagian[atr] == nilai).sum() + alfa) / (len(bagian)
                                                          + alfa * k)
        skor[c] = p                                              #<3>
    total = sum(skor.values())
    return " ".join(f"{c}: {s:.6f} -> {s / total:.4f}"
                    for c, s in skor.items())


x1 = {"tingkat": "perunggu", "kunjungan": "jarang",
      "keluhan": "ya", "promo": "tinggi"}
x2 = {"tingkat": "perunggu", "kunjungan": "jarang",
      "keluhan": "ya", "promo": "rendah"}
print("x1         ->", naive_bayes(mini, x1))
print("x2         ->", naive_bayes(mini, x2))
print("x2 Laplace ->", naive_bayes(mini, x2, alfa=1))           #<4>
