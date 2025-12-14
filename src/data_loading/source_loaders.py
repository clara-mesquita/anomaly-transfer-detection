"""
Data loaders for source datasets (labeled electrical fault datasets).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ElectricalFaultLoader:
    """
    Loader para o dataset Electrical Fault Detection and Classification.
    Este é o dataset fonte principal com 506 registros rotulados.
    
    Dataset: https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification
    Features: Voltage, Current, Temperature, Power Load, etc.
    Labels: Fault Type (Short Circuit, Overload, Ground Fault, etc.)
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: Caminho para o arquivo CSV. Se None, usa o caminho padrão.
        """
        if data_path is None:
            data_path = "data/raw/electrical_fault_detection.csv"
        self.data_path = Path(data_path)
        self._df_cache: Optional[pd.DataFrame] = None
        
    def load(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Carrega o dataset completo.
        
        Args:
            use_cache: Se True, usa cache em memória se disponível
            
        Returns:
            DataFrame com todas as colunas originais
        """
        if use_cache and self._df_cache is not None:
            logger.info("Usando cache do dataset de falhas elétricas")
            return self._df_cache.copy()
        
        logger.info(f"Carregando dataset de falhas elétricas de {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.data_path}\n"
                "Por favor, baixe o dataset do Kaggle:\n"
                "https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification\n"
                "E coloque em data/raw/electrical_fault_detection.csv"
            )
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Validar colunas esperadas
        expected_cols = ['Fault Type', 'Voltage (V)', 'Current (A)', 'Temperature (°C)']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Colunas esperadas não encontradas: {missing_cols}")
        
        self._df_cache = df.copy()
        return df
    
    def load_preprocessed(self, 
                         include_power_load: bool = True,
                         binary_labels: bool = True,
                         label_column: str = 'Component Health') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carrega e faz preprocessing básico do dataset.
        
        Args:
            include_power_load: Se True, inclui Power Load nas features
            binary_labels: Se True, converte labels para binário (0=normal, 1=anomalia)
            label_column: Coluna a ser usada como label ('Component Health' ou 'Fault Type')
            
        Returns:
            Tuple (features, labels) onde:
                - features: DataFrame com variáveis numéricas principais
                - labels: Series com labels (binária ou multiclass)
        """
        df = self.load()
        
        # Validar se a coluna de label existe
        if label_column not in df.columns:
            available_cols = [col for col in df.columns if 'health' in col.lower() or 'fault' in col.lower()]
            raise ValueError(
                f"Coluna '{label_column}' não encontrada no dataset.\n"
                f"Colunas disponíveis relacionadas: {available_cols}\n"
                f"Todas as colunas: {list(df.columns)}"
            )
        
        # Extrair features numéricas principais
        numeric_features = [
            'Voltage (V)',
            'Current (A)', 
            'Temperature (°C)'
        ]
        
        if include_power_load and 'Power Load (MW)' in df.columns:
            numeric_features.append('Power Load (MW)')
        
        X = df[numeric_features].copy()
        
        # Tratar valores faltantes se houver
        if X.isna().any().any():
            n_missing = X.isna().sum().sum()
            logger.warning(f"Encontrados {n_missing} valores faltantes, preenchendo com média")
            X = X.fillna(X.mean())
        
        # Processar labels baseado na coluna Component Health
        if binary_labels:
            if label_column == 'Component Health':
                # Component Health: Normal = 0 (não anomalia), Faulty/Overheated = 1 (anomalia)
                logger.info(f"Usando coluna '{label_column}' para labels")
                logger.info(f"Valores únicos encontrados: {df[label_column].unique()}")
                
                # Normal = 0, qualquer outra coisa (Faulty, Overheated) = 1
                y = (df[label_column] != 'Normal').astype(int)
                
                logger.info("Labels binárias (Component Health):")
                logger.info(f"  Normal (0): {sum(y==0)}")
                logger.info(f"  Anomalia (1): {sum(y==1)} [Faulty + Overheated]")
                
                # Mostrar distribuição detalhada
                for health_status in df[label_column].unique():
                    count = sum(df[label_column] == health_status)
                    logger.info(f"    - {health_status}: {count}")
                
            elif label_column == 'Fault Type':
                # Fallback para usar Fault Type se necessário
                anomaly_types = [
                    'Short Circuit', 
                    'Overload', 
                    'Ground Fault',
                    'Line Fault',
                    'Transformer Fault'
                ]
                y = df[label_column].isin(anomaly_types).astype(int)
                logger.info(f"Labels binárias (Fault Type): Normal={sum(y==0)}, Anomalia={sum(y==1)}")
            else:
                # Genérico: assume primeira categoria única é normal
                unique_vals = df[label_column].unique()
                normal_label = unique_vals[0]
                logger.warning(f"Coluna desconhecida, assumindo '{normal_label}' como normal")
                y = (df[label_column] != normal_label).astype(int)
                logger.info(f"Labels binárias: Normal={sum(y==0)}, Anomalia={sum(y==1)}")
        else:
            # Labels multiclass originais
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(df[label_column]), index=df.index)
            logger.info(f"Labels multiclass: {len(le.classes_)} classes")
            logger.info(f"Classes: {le.classes_}")
        
        logger.info(f"Features extraídas: {X.shape}")
        
        return X, y
    
    def get_feature_names(self, include_power_load: bool = True) -> List[str]:
        """
        Retorna os nomes das features numéricas principais.
        
        Args:
            include_power_load: Se True, inclui Power Load
            
        Returns:
            Lista de nomes de features
        """
        features = ['Voltage (V)', 'Current (A)', 'Temperature (°C)']
        if include_power_load:
            features.append('Power Load (MW)')
        return features
    
    def get_statistics(self) -> Dict:
        """
        Retorna estatísticas descritivas do dataset.
        
        Returns:
            Dict com estatísticas
        """
        df = self.load()
        numeric_cols = self.get_feature_names()
        
        stats = {
            'n_samples': len(df),
            'n_features': len(numeric_cols),
            'feature_stats': df[numeric_cols].describe().to_dict(),
            'fault_type_distribution': df['Fault Type'].value_counts().to_dict(),
            'missing_values': df[numeric_cols].isna().sum().to_dict()
        }
        
        return stats
    
    def get_info(self) -> Dict:
        """
        Retorna informações gerais sobre o dataset.
        
        Returns:
            Dict com metadados
        """
        df = self.load()
        
        # Verificar qual coluna de label está disponível
        label_info = {}
        if 'Component Health' in df.columns:
            label_info = {
                'label_column': 'Component Health',
                'label_values': df['Component Health'].value_counts().to_dict(),
                'label_type': 'multiclass (Normal, Faulty, Overheated)'
            }
        elif 'Fault Type' in df.columns:
            label_info = {
                'label_column': 'Fault Type',
                'label_values': df['Fault Type'].value_counts().to_dict(),
                'label_type': 'multiclass'
            }
        
        return {
            'name': 'Electrical Fault Detection and Classification',
            'source': 'Kaggle - E. Sathya Prakash',
            'url': 'https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification',
            'n_samples': len(df),
            'n_features': len(df.columns),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'has_labels': True,
            'temporal': False,
            'description': 'Electrical power system fault detection dataset with component health status',
            **label_info
        }

