import json
import sys
import pandas as pd
from fastapi.testclient import TestClient
sys.path.insert(0, "kode/bab27/aplikasi")
from api import app                                              #<1>

klien = TestClient(app)
sehat = klien.get("/sehat").json()
print("GET /sehat ->", sehat["status"], sehat["versi_model"],
      "AUC", sehat["auc_cv"])

d = pd.read_csv("kode/data/pelanggan_2026q3.csv").iloc[[0, 7]]   #<2>
d["lama_anggota_bln"] = (pd.Timestamp("2026-09-30") - pd.to_datetime(
    d.tanggal_daftar)).dt.days / 30.44
kolom = list(app.openapi()["components"]["schemas"]["Anggota"]
             ["properties"])
r = klien.post("/skor", json=json.loads(d[kolom].to_json(
    orient="records")))
print("POST /skor ->", r.status_code)
for h in r.json()["hasil"]:
    print(f"  {h['id_anggota']}: p_churn {h['p_churn']}, "
          f"kirim_voucer {h['kirim_voucer']}")

salah = json.loads(d[kolom].iloc[:1].to_json(orient="records"))  #<3>
salah[0]["porsi_promo"] = 1.7
r = klien.post("/skor", json=salah)
print("masukan salah ->", r.status_code,
      r.json()["detail"][0]["msg"])
