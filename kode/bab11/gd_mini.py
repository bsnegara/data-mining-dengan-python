import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([0.2, 0.5, 0.8, 1.0, 1.5, 2.5, 3.0, 4.5])            #<1>
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
X = np.c_[np.ones_like(x), x]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


b = np.zeros(2)                                                  #<2>
eta = 0.5
for it in range(1, 2001):
    p = sigmoid(X @ b)
    grad = X.T @ (p - y) / len(y)                                #<3>
    if it == 1:
        print("iterasi 1: p =", p.round(2), " gradien =", grad.round(4))
    b = b - eta * grad                                           #<4>
    if it in (1, 2, 10, 100, 2000):
        ll = np.sum(y * np.log(sigmoid(X @ b))
                    + (1 - y) * np.log(1 - sigmoid(X @ b)))
        print(f"iterasi {it:4d}: b0 = {b[0]:+.4f}, b1 = {b[1]:+.4f},"
              f" log-lik = {ll:.4f}")
sk = LogisticRegression(penalty=None).fit(x[:, None], y)         #<5>
print(f"scikit-learn: b0 = {sk.intercept_[0]:+.4f}, b1 = {sk.coef_[0, 0]:+.4f}")
