"""
Testes básicos para módulos de preprocessing.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.scalers import TransferScaler, handle_missing_values
from src.preprocessing.feature_alignment import FeatureAligner


def test_transfer_scaler_fit_transform():
    """Testa fit e transform do TransferScaler."""
    # Criar dados de exemplo
    X = np.random.randn(100, 3)
    
    # Testar standard scaling
    scaler = TransferScaler(method='standard')
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == X.shape
    assert scaler.is_fitted
    assert np.abs(X_scaled.mean(axis=0)).max() < 0.1  # Aproximadamente média 0
    assert np.abs(X_scaled.std(axis=0) - 1.0).max() < 0.1  # Aproximadamente std 1


def test_transfer_scaler_save_load(tmp_path):
    """Testa salvar e carregar scaler."""
    X = np.random.randn(50, 2)
    
    # Treinar e salvar
    scaler = TransferScaler(method='robust')
    scaler.fit(X)
    
    save_path = tmp_path / "scaler.pkl"
    scaler.save(save_path)
    
    # Carregar
    loaded_scaler = TransferScaler.load(save_path)
    
    # Verificar que transform produz mesmos resultados
    X_test = np.random.randn(10, 2)
    result1 = scaler.transform(X_test)
    result2 = loaded_scaler.transform(X_test)
    
    np.testing.assert_array_almost_equal(result1, result2)


def test_handle_missing_values():
    """Testa tratamento de valores faltantes."""
    # Criar DataFrame com missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, np.nan, 500]
    })
    
    # Testar estratégia mean
    df_filled = handle_missing_values(df, strategy='mean')
    assert df_filled.isna().sum().sum() == 0
    assert abs(df_filled['A'].iloc[2] - 3.0) < 0.01  # mean de [1,2,4,5] = 3
    
    # Testar estratégia zero
    df_zero = handle_missing_values(df, strategy='zero')
    assert df_zero.isna().sum().sum() == 0
    assert df_zero['B'].iloc[1] == 0


def test_feature_aligner_fit():
    """Testa alinhamento de features."""
    # Criar DataFrames de exemplo
    source_df = pd.DataFrame({
        'Voltage (V)': [220, 230, 240],
        'Current (A)': [10, 15, 20],
        'Temperature (°C)': [25, 30, 35]
    })
    
    target_df = pd.DataFrame({
        'Voltage': [225, 235],
        'Current': [12, 18],
        'Temperature': [28, 32],
        'Pressure': [100, 105]
    })
    
    # Criar aligner
    aligner = FeatureAligner()
    aligner.fit(source_df, target_df)
    
    # Verificar mapeamento
    assert len(aligner.common_features) == 3
    assert 'Voltage' in aligner.common_features
    assert 'Current' in aligner.common_features
    assert 'Temperature' in aligner.common_features


def test_feature_aligner_transform():
    """Testa transformação de features alinhadas."""
    source_df = pd.DataFrame({
        'Voltage (V)': [220, 230, 240],
        'Current (A)': [10, 15, 20]
    })
    
    target_df = pd.DataFrame({
        'Voltage': [225, 235],
        'Current': [12, 18]
    })
    
    aligner = FeatureAligner()
    aligner.fit(source_df, target_df)
    
    # Transform source to target format
    source_transformed = aligner.transform_source_to_target(source_df)
    
    assert 'Voltage' in source_transformed.columns
    assert 'Current' in source_transformed.columns
    assert 'Voltage (V)' not in source_transformed.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])