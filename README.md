Instruksi singkat:

1. Buat environment Python (3.8+ direkomendasikan) dan install dependensi:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Jalankan skrip untuk menghasilkan label dan melatih LSTM:

```bash
python bbca_lstm.py
```

Keterangan:
- Skrip mendeteksi pivot (swing points), mengelompokkan harga pivot dengan K-Means untuk menemukan level support/resistance,
  memberi label tiap bar apakah berdekatan dengan level tersebut, lalu membuat target pergerakan satu-hari ke depan
  (naik/turun) dan melatih LSTM sederhana.
- Ubah parameter `PIVOT_WINDOW`, `LOOKBACK`, atau `FUTURE_DAYS` langsung di `bbca_lstm.py` jika perlu.
