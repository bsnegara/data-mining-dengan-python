"""Layanan skor churn: uvicorn kode.bab27.aplikasi.api:app"""
import json
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FOLDER = Path(__file__).parents[1] / "model"                     #<1>
MODEL = joblib.load(FOLDER / "churn_v1.joblib")
META = json.loads((FOLDER / "churn_v1.json").read_text())
AMBANG = 0.278
app = FastAPI(title="Skor churn Kopi Nusantara", version=META["versi"])


class Anggota(BaseModel):                                        #<2>
    id_anggota: str
    kota: str
    tingkat: Literal["perunggu", "perak", "emas"]
    jenis_kelamin: Literal["L", "P"]
    umur: int = Field(ge=17, le=90)
    frekuensi_90h: int = Field(ge=0)
    frekuensi_90h_lalu: int = Field(ge=0)
    recency_hari: int = Field(ge=0)
    nilai_90h: float = Field(ge=0)
    rata_nilai_struk: float = Field(ge=0)
    porsi_promo: float = Field(ge=0, le=1)
    porsi_aplikasi: float = Field(ge=0, le=1)
    porsi_kopi_hitam: float = Field(ge=0, le=1)
    porsi_pagi: float = Field(ge=0, le=1)
    porsi_akhir_pekan: float = Field(ge=0, le=1)
    ragam_menu: int = Field(ge=0)
    keluhan_90h: int = Field(ge=0)
    jarak_km: float = Field(ge=0)
    pakai_ulasan: Literal[0, 1]
    lama_anggota_bln: float = Field(ge=0)


@app.post("/skor")                                               #<3>
def skor(daftar: list[Anggota]):
    X = pd.DataFrame([a.model_dump() for a in daftar])
    X["tren"] = X.frekuensi_90h / (X.frekuensi_90h_lalu + 1)
    p = MODEL.predict_proba(X[META["atribut"]])[:, 1]
    return {"versi_model": META["versi"],
            "hasil": [{"id_anggota": a.id_anggota,
                       "p_churn": round(float(q), 4),
                       "kirim_voucer": bool(q >= AMBANG)}
                      for a, q in zip(daftar, p)]}


@app.get("/sehat")                                               #<4>
def sehat():
    return {"status": "ok", "versi_model": META["versi"],
            "auc_cv": META["auc_cv"]}
