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


class VEDLoader:
    """
    Loader para o Vehicle Energy Dataset (VED).
    Dataset opcional para experimentos adicionais.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = "data/raw/ved/"
        self.data_path = Path(data_path)
        
    def load(self, file_name: str = "data.csv") -> pd.DataFrame:
        """
        Carrega um arquivo específico do VED.
        
        Args:
            file_name: Nome do arquivo dentro da pasta VED
        """
        file_path = self.data_path / file_name
        
        if not file_path.exists():
            logger.warning(f"Arquivo VED não encontrado: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        logger.info(f"VED dataset carregado: {df.shape}")
        
        return df
    
    def get_info(self) -> dict:
        """Retorna informações sobre o VED."""
        return {
            'name': 'Vehicle Energy Dataset (VED)',
            'source': 'GitHub - gsoh',
            'description': 'Real-world vehicle trip data with energy consumption',
            'has_labels': False
        }


def get_available_target_loaders() -> dict:
    """
    Retorna um dicionário com todos os loaders de datasets alvo disponíveis.
    
    Returns:
        Dict mapeando nome do dataset para sua classe loader
    """
    return {
        'can_bus': CANBusLoader,
        'ved': VEDLoader
    }


def compare_source_target_features(
    source_features: list, 
    target_features: list
) -> dict:
    """
    Compara features entre dataset fonte e alvo.
    
    Args:
        source_features: Lista de nomes de features do dataset fonte
        target_features: Lista de nomes de features do dataset alvo
        
    Returns:
        Dict com informações sobre alinhamento de features
    """
    # Normalizar nomes (remover unidades, converter para minúsculas)
    def normalize_name(name: str) -> str:
        return name.lower().replace('(', '').replace(')', '').replace(' ', '_')
    
    source_normalized = {normalize_name(f): f for f in source_features}
    target_normalized = {normalize_name(f): f for f in target_features}
    
    # Encontrar features em comum
    common_normalized = set(source_normalized.keys()) & set(target_normalized.keys())
    common_pairs = [
        (source_normalized[n], target_normalized[n]) 
        for n in common_normalized
    ]
    
    # Features únicas
    source_only = set(source_features) - {pair[0] for pair in common_pairs}
    target_only = set(target_features) - {pair[1] for pair in common_pairs}
    
    result = {
        'common_features': common_pairs,
        'n_common': len(common_pairs),
        'source_only': list(source_only),
        'target_only': list(target_only),
        'alignment_ratio': len(common_pairs) / max(len(source_features), 1)
    }
    
    logger.info(f"Alinhamento de features: {result['n_common']} em comum")
    logger.info(f"Apenas na fonte: {result['source_only']}")
    logger.info(f"Apenas no alvo: {result['target_only']}")
    
    return result