"""Dasbor retensi harian: streamlit run kode/bab27/aplikasi/dasbor.py"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retensi Kopi Nusantara", layout="wide")
skor = pd.read_parquet("keluaran/skor_harian.parquet")           #<1>
pantau = pd.read_csv("keluaran/pantau_mingguan.csv")

st.title("Program retensi anggota")                              #<2>
a, b, c = st.columns(3)
a.metric("Anggota aktif", f"{len(skor):,}".replace(",", "."))
b.metric("Disasar minggu ini", int(skor.kirim_voucer.sum()))
c.metric("Rata-rata skor churn", f"{skor.p_churn.mean():.1%}")

kota = st.multiselect("Kota", sorted(skor.kota.unique()))        #<3>
lihat = skor[skor.kota.isin(kota)] if kota else skor
st.bar_chart(lihat.groupby("kota").kirim_voucer.mean())
st.dataframe(lihat.nlargest(50, "p_churn")[
    ["id_anggota", "kota", "tingkat", "p_churn", "alasan_utama"]])

st.subheader("Kesehatan model")                                  #<4>
st.line_chart(pantau.set_index("minggu")[["psi_skor", "auc_kontrol"]])
if pantau.psi_skor.iloc[-1] > 0.25:
    st.error("Sebaran skor bergeser besar: jadwalkan pelatihan ulang.")
