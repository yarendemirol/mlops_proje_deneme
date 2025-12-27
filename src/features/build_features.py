import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

def apply_feature_engineering(df):
    # 1. Problem Reframing (Daha önce yaptığınız) [cite: 35]
    if 'Age' in df.columns:
        df['Age_Bucket'] = pd.cut(df['Age'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]).astype(int)
        print("[Kişi 2] Problem Reframing: Age Bucketized.")

    # 2. Feature Cross: Age_Bucket ve Male (Kullanıcı profili oluşturma)
    if 'Age_Bucket' in df.columns and 'Male' in df.columns:
        df['Age_Male_Cross'] = df['Age_Bucket'].astype(str) + "_" + df['Male'].astype(str)
        print("[Kişi 2] Feature Cross: Age and Male interactions created.")

    # 3. Hashed Feature: Ad Topic Line veya City gibi yüksek kardinalite için
    # Örnek olarak 'Ad Topic Line' sütununu hashleyelim (Eğer verinizde varsa)
    if 'Ad Topic Line' in df.columns:
        hasher = FeatureHasher(n_features=5, input_type='string')
        hashed_features = hasher.transform(df['Ad Topic Line'].apply(lambda x: [x]))
        hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'topic_hash_{i}' for i in range(5)])
        df = pd.concat([df.reset_index(drop=True), hashed_df], axis=1)
        print("[Kişi 2] Hashed Feature: High-cardinality topic lines transformed.")
    if 'Age_Male_Cross' in df.columns:
        # Get dummies ile string veriyi 0 ve 1'lere dönüştürürüz
        df = pd.get_dummies(df, columns=['Age_Male_Cross'], prefix='cross')
        print("[Kişi 2] One-Hot Encoding: Feature cross converted to numeric.")
    return df