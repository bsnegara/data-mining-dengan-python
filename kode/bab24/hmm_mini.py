import numpy as np
from hmmlearn.hmm import CategoricalHMM

pi = np.array([0.6, 0.4])                    # J = jelajah, N = niat #<1>
A = np.array([[0.7, 0.3],
              [0.2, 0.8]])
B = np.array([[0.6, 0.3, 0.1],               # simbol: m, d, k
              [0.2, 0.3, 0.5]])
o = [0, 1, 2]                                # m d k

alfa = np.zeros((3, 2))                                          #<2>
alfa[0] = pi * B[:, o[0]]
for t in range(1, 3):
    alfa[t] = (alfa[t - 1] @ A) * B[:, o[t]]
print("alfa:\n", alfa.round(5))
print("P(O) =", round(alfa[-1].sum(), 5))
print("P(N | m d k) =", round(alfa[-1, 1] / alfa[-1].sum(), 3))

beta = np.ones((3, 2))                                           #<3>
for t in (1, 0):
    beta[t] = A @ (B[:, o[t + 1]] * beta[t + 1])
gamma = alfa * beta / alfa[-1].sum()
print("beta:\n", beta.round(4))
print("gamma (posterior J, N):\n", gamma.round(3))

delta = np.zeros((3, 2)); asal = np.zeros((3, 2), int)           #<4>
delta[0] = pi * B[:, o[0]]
for t in range(1, 3):
    calon = delta[t - 1][:, None] * A
    asal[t] = calon.argmax(0)
    delta[t] = calon.max(0) * B[:, o[t]]
jalur = [int(delta[-1].argmax())]
for t in (2, 1):
    jalur.insert(0, int(asal[t, jalur[0]]))
print("delta:\n", delta.round(5))
print("jalur Viterbi:", "".join("JN"[j] for j in jalur),
      " peluang", round(delta[-1].max(), 5))

m = CategoricalHMM(n_components=2)                               #<5>
m.startprob_, m.transmat_, m.emissionprob_ = pi, A, B
X = np.array(o).reshape(-1, 1)
print("hmmlearn: log P(O) =", round(m.score(X), 4),
      "=", round(np.log(alfa[-1].sum()), 4))
logp, jl = m.decode(X, algorithm="viterbi")
print("hmmlearn Viterbi:", jl, round(np.exp(logp), 5))
