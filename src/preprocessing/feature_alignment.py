"""
Feature alignment utilities for source-target domain matching.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger


class FeatureAligner:
    """
    Classe para alinhar features entre domínios fonte e alvo.
    
    O alinhamento é necessário porque os datasets podem ter:
    - Nomes de colunas diferentes (ex: "Voltage (V)" vs "Voltage")
    - Features em ordens diferentes
    - Features adicionais que não existem em ambos os domínios
    """
    
    def __init__(self):
        """Inicializa o alinhador com mapeamentos vazios."""
        self.feature_mapping: Dict[str, str] = {}
        self.common_features: List[str] = []
        self.source_features: List[str] = []
        self.target_features: List[str] = []
        
    def fit(
        self, 
        source_df: pd.DataFrame, 
        target_df: pd.DataFrame,
        manual_mapping: Optional[Dict[str, str]] = None
    ) -> 'FeatureAligner':
        """
        Aprende o mapeamento de features entre fonte e alvo.
        
        Args:
            source_df: DataFrame do domínio fonte
            target_df: DataFrame do domínio alvo
            manual_mapping: Mapeamento manual opcional {source_col: target_col}
            
        Returns:
            self para method chaining
        """
        self.source_features = list(source_df.columns)
        self.target_features = list(target_df.columns)
        
        # Começar com mapeamento manual se fornecido
        if manual_mapping:
            self.feature_mapping = manual_mapping.copy()
            logger.info(f"Usando mapeamento manual: {manual_mapping}")
        else:
            # Tentar mapeamento automático baseado em normalização de nomes
            self.feature_mapping = self._auto_map_features(
                self.source_features, 
                self.target_features
            )
        
        # Identificar features comuns que podem ser usadas
        self.common_features = [
            target_feat 
            for source_feat, target_feat in self.feature_mapping.items()
            if source_feat in source_df.columns and target_feat in target_df.columns
        ]
        
        logger.info(f"Alinhamento concluído: {len(self.common_features)} features comuns")
        logger.info(f"Mapeamento: {self.feature_mapping}")
        
        return self
    
    def _auto_map_features(
        self, 
        source_cols: List[str], 
        target_cols: List[str]
    ) -> Dict[str, str]:
        """
        Cria mapeamento automático baseado em similaridade de nomes.
        
        Args:
            source_cols: Colunas do dataset fonte
            target_cols: Colunas do dataset alvo
            
        Returns:
            Dict mapeando colunas fonte para colunas alvo
        """
        mapping = {}
        
        # Normalizar nomes para comparação
        def normalize(name: str) -> str:
            # Remove unidades, espaços, converte para minúscula
            clean = name.lower()
            # Remove conteúdo entre parênteses (unidades)
            if '(' in clean:
                clean = clean[:clean.index('(')].strip()
            clean = clean.replace(' ', '').replace('_', '')
            return clean
        
        source_normalized = {normalize(col): col for col in source_cols}
        target_normalized = {normalize(col): col for col in target_cols}
        
        # Mapear features com nomes normalizados iguais
        for norm_name in source_normalized.keys():
            if norm_name in target_normalized:
                source_col = source_normalized[norm_name]
                target_col = target_normalized[norm_name]
                mapping[source_col] = target_col
        
        return mapping
    
    def transform_source_to_target(
        self, 
        source_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transforma DataFrame fonte para ter as mesmas colunas do alvo.
        
        Args:
            source_df: DataFrame fonte
            
        Returns:
            DataFrame com colunas renomeadas para match com alvo
        """
        if not self.feature_mapping:
            raise ValueError("Alinhador não foi fitted. Chame fit() primeiro.")
        
        # Selecionar apenas features mapeadas
        source_features_to_use = [
            col for col in source_df.columns 
            if col in self.feature_mapping
        ]
        
        df_aligned = source_df[source_features_to_use].copy()
        
        # Renomear colunas para nomes do alvo
        df_aligned = df_aligned.rename(columns=self.feature_mapping)
        
        return df_aligned
    
    def transform_target(
        self, 
        target_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transforma DataFrame alvo para conter apenas features comuns.
        
        Args:
            target_df: DataFrame alvo
            
        Returns:
            DataFrame com apenas colunas comuns
        """
        if not self.common_features:
            raise ValueError("Nenhuma feature comum encontrada após alinhamento.")
        
        # Selecionar apenas features comuns
        df_aligned = target_df[self.common_features].copy()
        
        return df_aligned
    
    def get_alignment_summary(self) -> Dict:
        """
        Retorna um resumo do alinhamento de features.
        
        Returns:
            Dict com estatísticas de alinhamento
        """
        return {
            'n_source_features': len(self.source_features),
            'n_target_features': len(self.target_features),
            'n_common_features': len(self.common_features),
            'common_features': self.common_features,
            'feature_mapping': self.feature_mapping,
            'alignment_ratio_source': len(self.common_features) / max(len(self.source_features), 1),
            'alignment_ratio_target': len(self.common_features) / max(len(self.target_features), 1)
        }


def create_default_alignment() -> Dict[str, str]:
    """
    Cria mapeamento padrão conhecido entre datasets elétricos comuns.
    
    Returns:
        Dict com mapeamento default {source_col: target_col}
    """
    # Mapeamento para Electrical Fault -> CAN Bus
    default_mapping = {
        'Voltage (V)': 'Voltage',
        'Current (A)': 'Current',
        'Temperature (°C)': 'Temperature'
    }
    
    return default_mapping


def align_data_distributions(
    source_data: np.ndarray,
    target_data: np.ndarray,
    method: str = 'standardize'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alinha distribuições de dados entre fonte e alvo.
    
    Isso é útil quando os ranges e escalas são muito diferentes,
    mas queremos que o modelo aprenda padrões transferíveis.
    
    Args:
        source_data: Array numpy de dados fonte
        target_data: Array numpy de dados alvo
        method: Método de alinhamento ('standardize', 'minmax', 'robust')
        
    Returns:
        Tuple (source_aligned, target_aligned)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    # Escolher scaler baseado no método
    if method == 'standardize':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    # Fit no fonte, transform em ambos
    source_aligned = scaler.fit_transform(source_data)
    target_aligned = scaler.transform(target_data)
    
    logger.info(f"Distribuições alinhadas usando {method}")
    
    return source_aligned, target_aligned