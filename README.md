# Workflow-CI

Repository untuk **Kriteria 3 - Membuat Workflow CI**, target level: **Skilled**.

## Struktur

```
Workflow-CI/
├── .github/workflows/
│   └── mlflow-ci.yml            # GitHub Actions workflow
└── MLProject/
    ├── MLProject                # MLflow Project spec
    ├── conda.yaml                # environment dependencies
    ├── modelling.py              # training script (dipanggil MLProject)
    └── namadataset_preprocessing/
        ├── train.csv             # ⚠️ kamu commit manual (lihat langkah di bawah)
        └── test.csv               # ⚠️ kamu commit manual
```

## Cara kerja workflow CI

File `.github/workflows/mlflow-ci.yml` akan berjalan otomatis setiap kali ada
`push` ke branch `main` yang menyentuh folder `MLProject/`, atau bisa dipicu
manual lewat tab **Actions > Run workflow** (`workflow_dispatch`).

Tahapan di dalam workflow:
1. **Checkout** repository
2. **Setup Python 3.12**
3. **Install MLflow** & dependencies (pandas, scikit-learn, dst)
4. **`mlflow run MLProject --env-manager=local`** → menjalankan `modelling.py`
   yang melatih ulang model dan mencatat run ke `mlruns/` (local file tracking)
5. **Simpan artefak** dengan dua cara sekaligus (memenuhi syarat Skilled):
   - Upload sebagai **GitHub Actions artifact** (bisa diunduh dari tab Actions run)
   - **Commit & push folder `mlruns/`** kembali ke repository GitHub yang sama

## Langkah setup sebelum push

1. Buat repository GitHub baru bernama `Workflow-CI`, visibility **public**.
2. Copy `train.csv` & `test.csv` hasil Kriteria 1 ke
   `MLProject/namadataset_preprocessing/`, lalu commit ke repo (lihat
   README di folder tersebut) — CI butuh datanya sudah ada di repo.
3. Push seluruh isi folder ini ke repo tersebut.
4. Di GitHub, buka **Settings > Actions > General > Workflow permissions**,
   pastikan **"Read and write permissions"** dipilih agar step commit/push
   di workflow bisa berhasil (karena workflow menggunakan `GITHUB_TOKEN`
   bawaan untuk push balik ke repo).
5. Push pertama akan otomatis memicu workflow. Cek tab **Actions** untuk
   memastikan run berhasil (centang hijau) minimal satu kali — ini syarat wajib.

## Ringkasan pemenuhan kriteria (Skilled)

| Syarat | Dipenuhi oleh |
|---|---|
| Folder `MLProject` berisi modelling.py, conda.yaml, MLProject | ✅ |
| Workflow CI via GitHub Actions yang melatih ulang model saat trigger | `.github/workflows/mlflow-ci.yml` |
| Artefak disimpan ke repositori (GitHub yang sama) | Step "Commit & push mlruns to repository" |

## Catatan submission
Setelah repo ini di-push dan minimal satu kali workflow run berhasil (centang
hijau di tab Actions), masukkan URL repo `Workflow-CI` ke file
`Workflow-CI.txt` pada folder submission utama (`SMSML_Lyly`).
