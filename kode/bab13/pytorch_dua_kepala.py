import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from siapkan import data_tensor

torch.manual_seed(0)
torch.set_num_threads(2)
df, X, G, y, latih, uji = data_tensor()


class Belanja(nn.Module):                                        #<1>
    def __init__(self, p, n_gerai=40, d_emb=3):
        super().__init__()
        self.emb = nn.Embedding(n_gerai, d_emb)                  #<2>
        self.badan = nn.Sequential(
            nn.Linear(p + d_emb, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU())
        self.kepala_beli = nn.Linear(32, 1)                      #<3>
        self.kepala_nilai = nn.Linear(32, 1)

    def forward(self, x, g):
        z = self.badan(torch.cat([x, self.emb(g)], dim=1))
        return (self.kepala_beli(z).squeeze(1),
                nn.functional.softplus(self.kepala_nilai(z)).squeeze(1))


def galat(logit, nilai, y):                                      #<4>
    beli = (y > 0).float()
    l1 = nn.functional.binary_cross_entropy_with_logits(logit, beli)
    l2 = (nn.functional.huber_loss(nilai, y, delta=50.0,
                                   reduction="none") * beli).sum() / beli.sum()
    return l1 + l2 / 100


model = Belanja(X.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
il, iv = latih[:4800], latih[4800:]                              #<5>
terbaik, sabar, simpan = np.inf, 0, None
for ep in range(1, 301):
    model.train()
    for i in torch.randperm(len(il)).split(128):                  #<6>
        b = torch.tensor(il)[i]
        opt.zero_grad()
        loss = galat(*model(X[b], G[b]), y[b])
        loss.backward()                                          #<7>
        opt.step()
    model.eval()
    with torch.no_grad():
        lv = galat(*model(X[iv], G[iv]), y[iv]).item()
    if lv < terbaik - 1e-4:
        terbaik, sabar = lv, 0
        simpan = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        sabar += 1
        if sabar == 20:                                          #<8>
            break
model.load_state_dict(simpan)
with torch.no_grad():
    logit, nilai = model(X[uji], G[uji])
    p = torch.sigmoid(logit)
    yhat = (p * nilai).numpy()                                   #<9>
yu = y[uji].numpy()
print(f"berhenti pada epoch {ep}, galat validasi terbaik {terbaik:.4f}")
print(f"MAE uji {np.abs(yu - yhat).mean():.1f} ribu Rp")
print(f"AUC 'akan belanja': {roc_auc_score(yu > 0, p):.3f}")
E = model.emb.weight.detach().numpy()
np.save("kode/bab13/prediksi_torch.npy", yhat)
np.savetxt("kode/bab13/embedding_gerai.txt", E, fmt="%.4f")
print("embedding gerai G01..G03:\n", E[:3].round(2))
