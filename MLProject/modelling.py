"""
modelling.py (versi CI)

Digunakan oleh MLflow Project (`MLProject`) untuk melakukan re-training model
secara otomatis setiap kali workflow CI dipicu.

Berbeda dari `modelling.py` pada folder `Membangun_model/`:
- Tracking URI menggunakan local file store (`file:./mlruns`), BUKAN
  `http://127.0.0.1:5000`, karena runner GitHub Actions tidak menjalankan
  MLflow Tracking Server secara live.
- Menerima argumen `--data_dir` agar path data fleksibel saat dijalankan
  lewat `mlflow run`.
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

TARGET_COL = "is_canceled"
RANDOM_STATE = 42


def load_data(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    return X_train, X_test, y_train, y_test


def main(data_dir: str):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Hotel_Booking_Cancellation_CI")
    mlflow.sklearn.autolog(log_models=True, log_model_signatures=True, log_input_examples=True)

    X_train, X_test, y_train, y_test = load_data(data_dir)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # PENTING: TIDAK diberi `run_name` di sini. Saat dijalankan lewat
    # `mlflow run .`, MLflow Projects CLI sudah membuat run tersendiri dan
    # meneruskan ID-nya lewat environment variable (MLFLOW_RUN_ID).
    # Jika start_run() diberi run_name/parameter lain, MLflow akan mencoba
    # membuat run BARU sehingga bentrok dengan run yang sudah disiapkan
    # (error: "active run ID does not match environment run ID").
    # Memanggil start_run() tanpa argumen membuatnya otomatis melanjutkan
    # run yang sudah ada tersebut.
    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        print(f"[CI] Test Accuracy : {acc:.4f}")
        print(f"[CI] Test F1-Score : {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="namadataset_preprocessing",
        help="Folder berisi train.csv dan test.csv"
    )
    args = parser.parse_args()
    main(args.data_dir)
