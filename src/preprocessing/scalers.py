"""
Scaling and normalization utilities for numerical features.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Union, Optional
import joblib
from pathlib import Path
from loguru import logger


class TransferScaler:
    """
    Scaler personalizado para cenários de transferência.
    
    Mantém estatísticas do domínio fonte e permite aplicar
    a mesma transformação no domínio alvo.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: Tipo de scaling ('standard', 'robust', 'minmax')
        """
        self.method = method
        self.scaler = None
        self.feature_names: Optional[list] = None
        self.is_fitted = False
        
        # Inicializar scaler apropriado
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'TransferScaler':
        """
        Aprende parâmetros de scaling dos dados fonte.
        
        Args:
            X: Dados de treino (fonte)
            
        Returns:
            self para method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
        
        self.scaler.fit(X_array)
        self.is_fitted = True
        
        logger.info(f"Scaler fitted com método {self.method}")
        logger.info(f"Shape: {X_array.shape}")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Aplica transformação de scaling.
        
        Args:
            X: Dados para transformar
            
        Returns:
            Dados transformados como numpy array
        """
        if not self.is_fitted:
            raise ValueError("Scaler precisa ser fitted antes de transform")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        return self.scaler.transform(X_array)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit e transform em uma operação.
        
        Args:
            X: Dados para fit e transform
            
        Returns:
            Dados transformados
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverte a transformação de scaling.
        
        Args:
            X: Dados transformados
            
        Returns:
            Dados na escala original
        """
        if not self.is_fitted:
            raise ValueError("Scaler precisa ser fitted")
        
        return self.scaler.inverse_transform(X)
    
    def save(self, path: Union[str, Path]):
        """
        Salva o scaler em disco.
        
        Args:
            path: Caminho para salvar (arquivo .pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, path)
        logger.info(f"Scaler salvo em {path}")
    
    @staticmethod
    def load(path: Union[str, Path]) -> 'TransferScaler':
        """
        Carrega scaler do disco.
        
        Args:
            path: Caminho do arquivo .pkl
            
        Returns:
            TransferScaler carregado
        """
        scaler = joblib.load(path)
        logger.info(f"Scaler carregado de {path}")
        return scaler
    
    def get_statistics(self) -> dict:
        """
        Retorna estatísticas do scaler.
        
        Returns:
            Dict com estatísticas (mean, std, etc. dependendo do método)
        """
        if not self.is_fitted:
            return {'fitted': False}
        
        stats = {'fitted': True, 'method': self.method}
        
        if hasattr(self.scaler, 'mean_'):
            stats['mean'] = self.scaler.mean_
        if hasattr(self.scaler, 'scale_'):
            stats['scale'] = self.scaler.scale_
        if hasattr(self.scaler, 'center_'):
            stats['center'] = self.scaler.center_
        if hasattr(self.scaler, 'data_min_'):
            stats['data_min'] = self.scaler.data_min_
        if hasattr(self.scaler, 'data_max_'):
            stats['data_max'] = self.scaler.data_max_
        
        return stats


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'mean',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Trata valores faltantes em um DataFrame.
    
    Args:
        df: DataFrame com possíveis valores faltantes
        strategy: Estratégia ('mean', 'median', 'zero', 'forward', 'constant')
        fill_value: Valor para usar se strategy='constant'
        
    Returns:
        DataFrame com valores faltantes tratados
    """
    df_filled = df.copy()
    
    if df_filled.isna().sum().sum() == 0:
        logger.info("Nenhum valor faltante encontrado")
        return df_filled
    
    n_missing = df_filled.isna().sum().sum()
    logger.info(f"Tratando {n_missing} valores faltantes com estratégia '{strategy}'")
    
    if strategy == 'mean':
        df_filled = df_filled.fillna(df_filled.mean())
    elif strategy == 'median':
        df_filled = df_filled.fillna(df_filled.median())
    elif strategy == 'zero':
        df_filled = df_filled.fillna(0)
    elif strategy == 'forward':
        df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("fill_value deve ser fornecido para strategy='constant'")
        df_filled = df_filled.fillna(fill_value)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    
    return df_filled


def remove_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers de um DataFrame.
    
    Args:
        df: DataFrame original
        method: Método de detecção ('iqr', 'zscore')
        threshold: Threshold para considerar outlier (1.5 para IQR, 3 para zscore)
        
    Returns:
        DataFrame sem outliers
    """
    df_clean = df.copy()
    n_original = len(df_clean)
    
    if method == 'iqr':
        # Método IQR (Interquartile Range)
        Q1 = df_clean.quantile(0.25)
        Q3 = df_clean.quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir limites
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Filtrar outliers
        mask = ~((df_clean < lower_bound) | (df_clean > upper_bound)).any(axis=1)
        df_clean = df_clean[mask]
        
    elif method == 'zscore':
        # Método Z-score
        z_scores = np.abs((df_clean - df_clean.mean()) / df_clean.std())
        mask = (z_scores < threshold).all(axis=1)
        df_clean = df_clean[mask]
        
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    n_removed = n_original - len(df_clean)
    logger.info(f"Outliers removidos: {n_removed} ({n_removed/n_original*100:.2f}%)")
    
    return df_clean