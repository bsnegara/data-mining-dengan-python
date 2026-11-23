from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory)
from normalisasi import KAMUS, STOP, bersihkan, stem

t = "Kopinya ENAK bgt, tp antrinya lama dan AC nya ga dingin di sini"
print("asli  :", t)                                               #<1>
print("norm  :", bersihkan(t, False, False, False))
print("+stop :", bersihkan(t, True, False, False))
print("+stem :", bersihkan(t, True, True, False))
print("+neg  :", bersihkan(t))

for w in ["pelayanannya", "pengirimannya", "kemanisan",           #<2>
          "kebanyakan", "diperbaiki", "baristanya", "menyenangkan",
          "berkualitas", "pegawainya", "kemahalan"]:
    print(f"  {w:>13s} -> {stem(w)}")

asli = set(StopWordRemoverFactory().get_stop_words())             #<3>
print(f"\nstopword Sastrawi: {len(asli)} kata")
print("dikeluarkan dari daftar:", ", ".join(sorted(asli - STOP)))
print("contoh kalimat negatif dengan stopword bawaan:")
print("  ", " ".join(w for w in "kopinya tidak enak tapi murah"
                     .split() if w not in asli))
