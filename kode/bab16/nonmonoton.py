import numpy as np
from scipy.cluster.hierarchy import linkage

X = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.8]])               #<1>
for metode in ["centroid", "single", "complete", "average", "ward"]:
    Z = linkage(X, metode)
    print(f"{metode:9s} tinggi gabungan: {Z[0, 2]:.3f} lalu {Z[1, 2]:.3f}"
          + ("   <- turun (inversi)" if Z[1, 2] < Z[0, 2] else ""))
