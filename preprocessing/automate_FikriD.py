"""
automate_NamaAnda.py
====================
Script otomatisasi pipeline preprocessing untuk dataset Credit Risk.
Mengonversi seluruh tahapan eksperimen dari notebook menjadi fungsi
yang dapat dijalankan secara mandiri maupun dari CI/CD workflow.

Penggunaan:
    python automate_NamaAnda.py \
        --input  ../credit_risk_dataset_raw.csv \
        --output credit_risk_preprocessing
"""

import os
import argparse
import logging
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Konfigurasi logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Konstanta ──────────────────────────────────────────────────────────────────
TARGET_COL   = "loan_status"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
IQR_FACTOR   = 1.5


# ──────────────────────────────────────────────────────────────────────────────
# Fungsi-fungsi preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV ke dalam DataFrame.

    Parameters
    ----------
    path : str
        Jalur lengkap menuju file CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame yang berisi data mentah.

    Raises
    ------
    FileNotFoundError
        Apabila file tidak ditemukan pada path yang diberikan.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)
    log.info("Data berhasil dimuat → %d baris, %d kolom", *df.shape)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame masukan.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa baris duplikat.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed:
        log.info("Duplikat dihapus: %d baris", n_removed)
    else:
        log.info("Tidak ada duplikat ditemukan.")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengisi nilai yang hilang (missing values).

    Strategi:
      - Fitur numerik  → median (robust terhadap outlier).
      - Fitur kategori → modus (nilai terbanyak).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang mungkin mengandung missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa missing values.
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing:
            fill_val = df[col].median()
            df[col].fillna(fill_val, inplace=True)
            log.info("[MEDIAN] %s → %d nilai diisi (%.4f)", col, n_missing, fill_val)

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        n_missing = df[col].isnull().sum()
        if n_missing:
            fill_val = df[col].mode()[0]
            df[col].fillna(fill_val, inplace=True)
            log.info("[MODUS ] %s → %d nilai diisi (%s)", col, n_missing, fill_val)

    total_remaining = df.isnull().sum().sum()
    log.info("Total missing setelah imputasi: %d", total_remaining)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah fitur kategori menjadi representasi numerik
    menggunakan Label Encoding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan fitur kategori bertipe object.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur kategori telah diencode.
    """
    df = df.copy()
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        log.info("Label Encoding selesai: %s", col)

    log.info("Total kolom setelah encoding: %d", df.shape[1])
    return df


def cap_outliers(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Menangani outlier dengan metode IQR Capping (Winsorizing).

    Nilai di luar batas [Q1 - k*IQR, Q3 + k*IQR] dipotong ke nilai
    batas tersebut sehingga tidak ada data yang dihapus.

    Parameters
    ----------
    df     : pd.DataFrame  DataFrame yang akan diproses.
    target : str           Nama kolom target yang dilewati.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan outlier yang sudah di-capping.
    """
    df = df.copy()
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target
    ]

    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - IQR_FACTOR * IQR
        upper = Q3 + IQR_FACTOR * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    log.info("Outlier capping selesai pada %d fitur numerik.", len(feature_cols))
    return df


def standardize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menstandarisasi fitur menggunakan StandardScaler.

    Scaler di-fit hanya pada data training untuk menghindari data leakage,
    kemudian diterapkan pada data testing.

    Parameters
    ----------
    X_train : pd.DataFrame  Data fitur training.
    X_test  : pd.DataFrame  Data fitur testing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (X_train_scaled, X_test_scaled)
    """
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    log.info("Standarisasi selesai. Mean fitur training ≈ 0, Std ≈ 1.")
    return X_train_sc, X_test_sc


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Memisahkan dataset menjadi set training dan testing secara stratified.

    Stratified split memastikan proporsi kelas pada kedua set
    seimbang dengan distribusi kelas pada dataset asli.

    Parameters
    ----------
    df           : pd.DataFrame  DataFrame yang sudah diproses.
    target       : str           Nama kolom target.
    test_size    : float         Proporsi data testing (default 0.20).
    random_state : int           Seed acak untuk reprodusibilitas.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    log.info(
        "Split selesai → Train: %d | Test: %d",
        X_train.shape[0], X_test.shape[0],
    )
    return X_train, X_test, y_train, y_test


def save_results(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """
    Menyimpan hasil preprocessing ke dalam file CSV.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame  Data fitur.
    y_train, y_test : pd.Series     Data target.
    output_dir      : str           Folder penyimpanan output.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"),  index=False)

    log.info("Hasil preprocessing disimpan ke: %s/", output_dir)
    for fname in os.listdir(output_dir):
        fsize = os.path.getsize(os.path.join(output_dir, fname)) / 1024
        log.info("  ↳ %-20s (%.1f KB)", fname, fsize)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline utama
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing(input_path: str, output_dir: str) -> None:
    """
    Menjalankan pipeline preprocessing end-to-end.

    Urutan tahapan:
        1. Muat data mentah
        2. Hapus duplikat
        3. Imputasi missing values
        4. Encoding fitur kategorik
        5. Capping outlier (IQR)
        6. Train-Test Split (stratified)
        7. Standarisasi fitur
        8. Simpan hasil

    Parameters
    ----------
    input_path : str  Path ke file CSV data mentah.
    output_dir : str  Folder tujuan penyimpanan hasil preprocessing.
    """
    log.info("=" * 55)
    log.info("   PIPELINE PREPROCESSING CREDIT RISK DATASET")
    log.info("=" * 55)

    # Tahap 1 – Muat data
    df = load_data(input_path)

    # Tahap 2 – Hapus duplikat
    df = remove_duplicates(df)

    # Tahap 3 – Imputasi missing values
    df = impute_missing(df)

    # Tahap 4 – Encoding kategori
    df = encode_categoricals(df)

    # Tahap 5 – Capping outlier
    df = cap_outliers(df, target=TARGET_COL)

    # Tahap 6 – Split data
    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COL)

    # Tahap 7 – Standarisasi
    X_train, X_test = standardize_features(X_train, X_test)

    # Tahap 8 – Simpan
    save_results(X_train, X_test, y_train, y_test, output_dir)

    log.info("=" * 55)
    log.info("   PREPROCESSING SELESAI")
    log.info("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate preprocessing pipeline untuk Credit Risk dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../credit_risk_dataset_raw.csv",
        help="Path ke file CSV data mentah.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="credit_risk_preprocessing",
        help="Folder tujuan output hasil preprocessing.",
    )
    args = parser.parse_args()
    run_preprocessing(args.input, args.output)
