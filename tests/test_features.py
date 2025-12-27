import pandas as pd
import numpy as np
from src.features.build_features import apply_feature_engineering


def test_apply_feature_engineering():
    # Test verisi oluştur
    test_df = pd.DataFrame({'Age': [20, 0, 40]})

    # Fonksiyonu çalıştır
    result_df = apply_feature_engineering(test_df)

    # Kontroller (Assertions)
    assert 'Age_Bucket' in result_df.columns  # Kolon oluştu mu?
    assert result_df['Age_Bucket'].iloc[0] == 2  # 20 yaş doğru bucket'ta mı?
    assert not result_df['Age_Bucket'].isnull().any()  # Boş değer var mı?
    print("\n[SUCCESS] Feature Engineering Unit Testi başarıyla geçti.")