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
    # CLI). CLI menyiapkan run lewat environment variable MLFLOW_RUN_ID (plus
    # MLFLOW_EXPERIMENT_ID & MLFLOW_TRACKING_URI) SEBELUM script dieksekusi.
    # `mlflow.active_run()` TIDAK mendeteksi ini (masih None sampai
    # start_run() benar-benar dipanggil) - makanya deteksi harus lewat env
    # var secara langsung.
    #
    # Kalau dipanggil via `mlflow run`: JANGAN override tracking URI/experiment
    # secara manual - biarkan environment variable dari CLI yang menentukan,
    # supaya start_run() otomatis melanjutkan run yang sudah disiapkan.
    #
    # Kalau dipanggil langsung (`python modelling.py`): tidak ada env var itu,
    # jadi kita atur sendiri tracking URI & experiment-nya.
    running_via_mlflow_cli = "MLFLOW_RUN_ID" in os.environ

    if not running_via_mlflow_cli:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Hotel_Booking_Cancellation_CI")

    X_train, X_test, y_train, y_test = load_data(data_dir)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

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
