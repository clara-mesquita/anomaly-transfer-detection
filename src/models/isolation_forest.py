"""
Isolation Forest model for anomaly detection.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as SKLearnIsolationForest
from typing import Optional, Dict, Union
import joblib
from pathlib import Path
from loguru import logger


class IsolationForestDetector:
    """
    Wrapper para Isolation Forest com funcionalidades adicionais
    para transferência entre domínios.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, str] = 'auto',
        contamination: float = 0.1,
        max_features: float = 1.0,
        random_state: int = 42
    ):
        """
        Args:
            n_estimators: Número de árvores na floresta
            max_samples: Número de amostras para treinar cada árvore
            contamination: Proporção esperada de anomalias nos dados
            max_features: Número de features para considerar em cada split
            random_state: Seed para reprodutibilidade
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        
        self.model = None
        self.is_fitted = False
        self.feature_names: Optional[list] = None
        self.decision_threshold: Optional[float] = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Treina o modelo Isolation Forest.
        
        Note que Isolation Forest é não-supervisionado, então y é ignorado.
        Mas mantemos o parâmetro para compatibilidade com sklearn API.
        
        Args:
            X: Features de treino
            y: Labels (opcional, ignorado)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
        
        # Inicializar modelo
        self.model = SKLearnIsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1  # Usar todos os cores disponíveis
        )
        
        # Treinar
        logger.info("Treinando Isolation Forest...")
        self.model.fit(X_array)
        
        # Calcular threshold de decisão baseado nos scores de treino
        scores = self.model.decision_function(X_array)
        self.decision_threshold = np.percentile(scores, self.contamination * 100)
        
        self.is_fitted = True
        logger.info(f"Isolation Forest treinado com {X_array.shape[0]} amostras")
        logger.info(f"Decision threshold: {self.decision_threshold:.4f}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Prediz se cada amostra é anomalia (1) ou normal (0).
        
        Args:
            X: Features para predição
            
        Returns:
            Array com 1 para anomalia, 0 para normal
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # sklearn retorna -1 para anomalia, 1 para normal
        # Convertemos para 1=anomalia, 0=normal
        predictions = self.model.predict(X_array)
        predictions = np.where(predictions == -1, 1, 0)
        
        return predictions
    
    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Retorna scores de anomalia (quanto menor, mais anômalo).
        
        Args:
            X: Features para scoring
            
        Returns:
            Array com scores de anomalia
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        return self.model.decision_function(X_array)
    
    def score_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Alias para decision_function com interpretação invertida.
        Retorna scores onde valores MAIORES indicam maior anomalia.
        
        Args:
            X: Features para scoring
            
        Returns:
            Array com scores de anomalia (maior = mais anômalo)
        """
        # decision_function retorna valores negativos para anomalias
        # Invertemos o sinal para que maior = mais anômalo
        return -self.decision_function(X)
    
    def set_contamination(self, contamination: float):
        """
        Ajusta o nível de contaminação (threshold) sem retreinar.
        
        Útil para transferência: podemos treinar no fonte e ajustar
        threshold no alvo.
        
        Args:
            contamination: Nova proporção de contaminação (0 a 0.5)
        """
        if not self.is_fitted:
            raise ValueError("Modelo precisa estar treinado")
        
        self.contamination = contamination
        self.model.contamination = contamination
        
        logger.info(f"Contaminação ajustada para {contamination}")
    
    def save(self, path: Union[str, Path]):
        """
        Salva o modelo treinado.
        
        Args:
            path: Caminho para salvar (.pkl ou .joblib)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar tudo: modelo + metadados
        save_dict = {
            'model': self.model,
            'feature_names': self.feature_names,
            'decision_threshold': self.decision_threshold,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"Modelo salvo em {path}")
    
    @staticmethod
    def load(path: Union[str, Path]) -> 'IsolationForestDetector':
        """
        Carrega modelo do disco.
        
        Args:
            path: Caminho do arquivo
            
        Returns:
            IsolationForestDetector carregado
        """
        save_dict = joblib.load(path)
        
        # Reconstruir detector
        detector = IsolationForestDetector(
            n_estimators=save_dict.get('n_estimators', 100),
            contamination=save_dict.get('contamination', 0.1)
        )
        
        detector.model = save_dict['model']
        detector.feature_names = save_dict.get('feature_names')
        detector.decision_threshold = save_dict.get('decision_threshold')
        detector.is_fitted = True
        
        logger.info(f"Modelo carregado de {path}")
        return detector
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Retorna importâncias aproximadas das features.
        
        Para Isolation Forest, não há feature importance direta,
        mas podemos aproximar baseado na profundidade média.
        
        Returns:
            Array com scores de importância (ou None se não disponível)
        """
        # Isolation Forest não tem feature_importances_ nativo
        # Retornamos None ou implementaríamos uma aproximação
        logger.warning("Isolation Forest não fornece feature importances diretas")
        return None
    
    def get_params(self) -> Dict:
        """
        Retorna parâmetros do modelo.
        
        Returns:
            Dict com hiperparâmetros
        """
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'decision_threshold': self.decision_threshold
        }