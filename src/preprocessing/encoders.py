"""
Encoding utilities for categorical variables.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Dict, List, Optional, Union
import joblib
from pathlib import Path
from loguru import logger


class CategoricalEncoder:
    """
    Encoder para variáveis categóricas com suporte a transferência.
    Mantém mapeamento consistente entre domínios.
    """
    
    def __init__(self, method: str = 'label'):
        """
        Args:
            method: Método de encoding ('label', 'onehot')
        """
        self.method = method
        self.encoders: Dict[str, Union[LabelEncoder, OneHotEncoder]] = {}
        self.is_fitted = False
        self.categorical_columns: List[str] = []
        
    def fit(self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None):
        """
        Aprende encoding das colunas categóricas.
        
        Args:
            df: DataFrame com variáveis categóricas
            categorical_columns: Lista de colunas categóricas. Se None, detecta automaticamente.
        """
        if categorical_columns is None:
            # Detectar automaticamente colunas categóricas
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.categorical_columns = categorical_columns
        
        for col in categorical_columns:
            if col not in df.columns:
                logger.warning(f"Coluna {col} não encontrada no DataFrame")
                continue
            
            if self.method == 'label':
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.encoders[col] = encoder
                
            elif self.method == 'onehot':
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoder.fit(df[[col]].astype(str))
                self.encoders[col] = encoder
                
            else:
                raise ValueError(f"Método desconhecido: {self.method}")
        
        self.is_fitted = True
        logger.info(f"Encoder fitted para {len(self.encoders)} colunas com método '{self.method}'")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica encoding às colunas categóricas.
        
        Args:
            df: DataFrame para transformar
            
        Returns:
            DataFrame com colunas categóricas encodadas
        """
        if not self.is_fitted:
            raise ValueError("Encoder não foi fitted")
        
        df_encoded = df.copy()
        
        for col, encoder in self.encoders.items():
            if col not in df_encoded.columns:
                logger.warning(f"Coluna {col} não encontrada, pulando")
                continue
            
            if self.method == 'label':
                # Label encoding
                try:
                    df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                except ValueError as e:
                    # Valores desconhecidos - mapear para -1
                    logger.warning(f"Valores desconhecidos em {col}, mapeando para -1")
                    known_values = set(encoder.classes_)
                    df_encoded[col] = df_encoded[col].astype(str).apply(
                        lambda x: encoder.transform([x])[0] if x in known_values else -1
                    )
            
            elif self.method == 'onehot':
                # One-hot encoding
                encoded_array = encoder.transform(df_encoded[[col]].astype(str))
                
                # Criar nomes das novas colunas
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # Adicionar colunas encodadas
                for i, feature_name in enumerate(feature_names):
                    df_encoded[feature_name] = encoded_array[:, i]
                
                # Remover coluna original
                df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded
    
    def fit_transform(self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit e transform em uma operação.
        
        Args:
            df: DataFrame para fit e transform
            categorical_columns: Colunas categóricas
            
        Returns:
            DataFrame transformado
        """
        return self.fit(df, categorical_columns).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nomes das features após encoding.
        
        Returns:
            Lista de nomes de features
        """
        if not self.is_fitted:
            return []
        
        feature_names = []
        
        if self.method == 'label':
            feature_names = self.categorical_columns
        
        elif self.method == 'onehot':
            for col, encoder in self.encoders.items():
                for cat in encoder.categories_[0]:
                    feature_names.append(f"{col}_{cat}")
        
        return feature_names
    
    def save(self, path: Union[str, Path]):
        """Salva o encoder."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Encoder salvo em {path}")
    
    @staticmethod
    def load(path: Union[str, Path]) -> 'CategoricalEncoder':
        """Carrega encoder do disco."""
        encoder = joblib.load(path)
        logger.info(f"Encoder carregado de {path}")
        return encoder


def encode_fault_types(fault_type_series: pd.Series) -> pd.Series:
    """
    Codifica tipos de falhas em labels binárias (anomalia ou não).
    
    Específico para o dataset Electrical Fault Detection.
    
    Args:
        fault_type_series: Series com tipos de falhas
        
    Returns:
        Series binária (0=normal, 1=anomalia)
    """
    # Falhas consideradas anomalias
    anomaly_types = ['Short Circuit', 'Overload', 'Ground Fault']
    
    binary_labels = fault_type_series.isin(anomaly_types).astype(int)
    
    logger.info(f"Fault types encodados: {np.bincount(binary_labels)}")
    
    return binary_labels