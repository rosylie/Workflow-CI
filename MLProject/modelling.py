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
    mlflow.sklearn.autolog(log_models=True, log_model_signatures=True, log_input_examples=True)

    # Deteksi apakah script ini dipanggil lewat `mlflow run .` (MLflow Projects
    # CLI). Jika ya, MLflow CLI SUDAH menyiapkan tracking URI, experiment, dan
    # run aktif lewat environment variable sebelum script ini dieksekusi.
    # Memanggil mlflow.set_experiment() / mlflow.start_run() secara manual di
    # kondisi ini akan BENTROK dengan konteks yang sudah disiapkan CLI
    # (error: "active run ID does not match environment run ID").
    #
    # Jadi: kalau sudah ada active run (dipanggil via `mlflow run`), langsung
    # training saja tanpa start_run/set_experiment tambahan. Kalau tidak ada
    # active run (dipanggil langsung via `python modelling.py`), baru kita
    # atur sendiri tracking URI, experiment, dan run-nya.
    active_run = mlflow.active_run()
    started_run_here = False

    if active_run is None:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Hotel_Booking_Cancellation_CI")
        mlflow.start_run()
        started_run_here = True

    X_train, X_test, y_train, y_test = load_data(data_dir)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    try:
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        print(f"[CI] Test Accuracy : {acc:.4f}")
        print(f"[CI] Test F1-Score : {f1:.4f}")
    finally:
        if started_run_here:
            mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="namadataset_preprocessing",
        help="Folder berisi train.csv dan test.csv"
    )
    args = parser.parse_args()
    main(args.data_dir)
