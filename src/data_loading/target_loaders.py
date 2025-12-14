"""
Data loaders for target datasets (vehicle-related, mostly unlabeled).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


class CANBusLoader:
    """
    Loader para o CAN Bus Anomaly Detection Dataset.
    Este é o dataset alvo principal com 46,623 registros de dados veiculares.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: Caminho para o arquivo CSV. Se None, usa o caminho padrão.
        """
        if data_path is None:
            data_path = "data/raw/can_bus_anomaly_detection.csv"
        self.data_path = Path(data_path)
        
    def load(self) -> pd.DataFrame:
        """
        Carrega o dataset completo.
        
        Returns:
            DataFrame com todas as colunas originais
        """
        logger.info(f"Carregando dataset CAN Bus de {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.data_path}. "
                "Por favor, baixe do Kaggle ou coloque na pasta data/raw/"
            )
        
        df = pd.read_csv(self.data_path)
        logger.info(f"CAN Bus dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        return df
    
    def load_preprocessed(self, parse_datetime: bool = True) -> pd.DataFrame:
        """
        Carrega e faz preprocessing básico do dataset.
        
        Args:
            parse_datetime: Se True, converte coluna datetime para tipo datetime
            
        Returns:
            DataFrame com preprocessing básico aplicado
        """
        df = self.load()
        
        # Parse datetime se solicitado
        if parse_datetime and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            logger.info("Coluna datetime convertida para tipo datetime")
        
        # Ordenar por tempo se datetime existe
        if 'datetime' in df.columns and df['datetime'].notna().any():
            df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def get_aligned_features(self) -> pd.DataFrame:
        """
        Retorna apenas as features que podem ser alinhadas com o dataset fonte.
        
        Features comuns: Voltage, Current, Temperature
        
        Returns:
            DataFrame com features alinhadas
        """
        df = self.load_preprocessed()
        
        # Features que existem tanto no fonte quanto no alvo
        aligned_features = ['Voltage', 'Current', 'Temperature']
        
        # Verificar quais features existem
        available = [f for f in aligned_features if f in df.columns]
        
        if len(available) < len(aligned_features):
            missing = set(aligned_features) - set(available)
            logger.warning(f"Features ausentes no dataset alvo: {missing}")
        
        X = df[available].copy()
        logger.info(f"Features alinhadas extraídas: {X.shape}")
        
        return X
    
    def has_labels(self) -> bool:
        """
        Verifica se o dataset tem labels (coluna 'tag').
        
        Returns:
            True se a coluna 'tag' existe e contém variação
        """
        df = self.load()
        if 'tag' not in df.columns:
            return False
        
        # Se 'tag' tem mais de um valor único, provavelmente são labels
        return df['tag'].nunique() > 1
    
    def get_labels(self) -> Optional[pd.Series]:
        """
        Retorna labels se disponíveis.
        
        Returns:
            Series com labels ou None se não houver
        """
        if not self.has_labels():
            logger.warning("Dataset CAN Bus não possui labels claros")
            return None
        
        df = self.load()
        return df['tag']
    
    def get_feature_names(self) -> list:
        """
        Retorna os nomes das features disponíveis.
        """
        df = self.load()
        # Excluir colunas não-numéricas ou de identificação
        exclude = ['tag', 'datetime']
        return [col for col in df.columns if col not in exclude]
    
    def get_info(self) -> dict:
        """
        Retorna informações sobre o dataset.
        """
        df = self.load()
        return {
            'name': 'CAN Bus Anomaly Detection Dataset',
            'source': 'Kaggle - Ankit Sharma',
            'n_samples': len(df),
            'n_features': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'has_labels': self.has_labels(),
            'label_column': 'tag' if self.has_labels() else None,
            'temporal': 'datetime' in df.columns
        }

