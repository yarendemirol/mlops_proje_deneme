import numpy as np
import pandas as pd


def apply_feature_engineering(df):
    """Döküman III.1: Problem Reframing [cite: 35]"""
    if 'Age' in df.columns:
        # Önce boş değerleri dolduruyoruz ki astype(int) hata vermesin
        df['Age'] = df['Age'].fillna(df['Age'].median())

        # Senin doğru dediğin hesaplama mantığı [cite: 35]
        df['Age_Bucket'] = pd.cut(df['Age'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]).astype(int)
        print("[Kişi 2] Problem Reframing: Age Bucketized.")
    return df