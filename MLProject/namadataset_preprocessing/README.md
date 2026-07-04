# namadataset_preprocessing

Letakkan `train.csv` dan `test.csv` di sini (hasil Kriteria 1), lalu **commit &
push file-nya ke repository** ini sebelum menjalankan workflow CI.

Ini penting: runner GitHub Actions tidak punya akses ke Google Drive/Kaggle
secara langsung, jadi data siap-latih harus sudah ada di dalam repo agar
`mlflow run` di CI bisa membacanya.

```bash
cp ../../Eksperimen_SML_Lyly/preprocessing/namadataset_preprocessing/*.csv .
git add train.csv test.csv
git commit -m "add preprocessed dataset for CI"
git push
```
