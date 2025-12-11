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
                         binary_labels: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carrega e faz preprocessing básico do dataset.
        
        Args:
            include_power_load: Se True, inclui Power Load nas features
            binary_labels: Se True, converte labels para binário (0=normal, 1=anomalia)
            
        Returns:
            Tuple (features, labels) onde:
                - features: DataFrame com variáveis numéricas principais
                - labels: Series com labels (binária ou multiclass)
        """
        df = self.load()
        
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
        
        # Processar labels
        if binary_labels:
            # Criar label binária: considerar tipos específicos como anomalias
            # Short Circuit, Overload, Ground Fault são claramente anomalias
            # Outros tipos podem ser operação normal ou manutenção
            anomaly_types = [
                'Short Circuit', 
                'Overload', 
                'Ground Fault',
                'Line Fault',
                'Transformer Fault'
            ]
            y = df['Fault Type'].isin(anomaly_types).astype(int)
            logger.info(f"Labels binárias: Normal={sum(y==0)}, Anomalia={sum(y==1)}")
        else:
            # Labels multiclass originais
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(df['Fault Type']), index=df.index)
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
        return {
            'name': 'Electrical Fault Detection and Classification',
            'source': 'Kaggle - E. Sathya Prakash',
            'url': 'https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification',
            'n_samples': len(df),
            'n_features': len(df.columns),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'has_labels': True,
            'label_column': 'Fault Type',
            'label_type': 'multiclass',
            'temporal': False,
            'description': 'Electrical power system fault detection dataset with various fault types'
        }


class SKABLoader:
    """
    Loader para o Skoltech Anomaly Benchmark (SKAB).
    Dataset de sistema de circulação de água com anomalias anotadas.
    
    Dataset: https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab
    Features: Sensores diversos (pressão, temperatura, vibração, etc.)
    Labels: Binary (0=normal, 1=anomaly)
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: Caminho para a pasta do SKAB. Se None, usa o caminho padrão.
        """
        if data_path is None:
            data_path = "data/raw/skab/"
        self.data_path = Path(data_path)
        
    def load(self, file_name: str = "data.csv") -> pd.DataFrame:
        """
        Carrega um arquivo específico do SKAB.
        
        O SKAB contém múltiplos arquivos CSV, cada um representando
        um experimento diferente com diferentes tipos de anomalias.
        
        Args:
            file_name: Nome do arquivo CSV dentro da pasta SKAB
            
        Returns:
            DataFrame com dados do experimento
        """
        file_path = self.data_path / file_name
        
        if not file_path.exists():
            logger.warning(f"Arquivo SKAB não encontrado: {file_path}")
            logger.info("Arquivos disponíveis na pasta:")
            if self.data_path.exists():
                for f in self.data_path.glob("*.csv"):
                    logger.info(f"  - {f.name}")
            return pd.DataFrame()
        
        # SKAB usa separador ; e vírgula como decimal
        df = pd.read_csv(file_path, sep=';', decimal=',', index_col='datetime', parse_dates=True)
        logger.info(f"SKAB dataset carregado: {df.shape}")
        
        return df
    
    def load_all(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Carrega todos os arquivos CSV disponíveis no diretório SKAB.
        
        Returns:
            Lista de tuplas (nome_arquivo, dataframe)
        """
        if not self.data_path.exists():
            logger.error(f"Diretório SKAB não encontrado: {self.data_path}")
            return []
        
        datasets = []
        for csv_file in self.data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, sep=';', decimal=',', 
                               index_col='datetime', parse_dates=True)
                datasets.append((csv_file.stem, df))
                logger.info(f"Carregado: {csv_file.name} - {df.shape}")
            except Exception as e:
                logger.warning(f"Erro ao carregar {csv_file.name}: {e}")
        
        logger.info(f"Total de {len(datasets)} arquivos SKAB carregados")
        return datasets
    
    def load_preprocessed(self, file_name: str = "data.csv") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carrega e preprocessa um arquivo SKAB.
        
        Args:
            file_name: Nome do arquivo
            
        Returns:
            Tuple (features, labels)
        """
        df = self.load(file_name)
        
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        # Separar features e labels
        # SKAB tem coluna 'anomaly' (0=normal, 1=anomaly) e 'changepoint'
        if 'anomaly' in df.columns:
            y = df['anomaly'].astype(int)
            X = df.drop(columns=['anomaly', 'changepoint'], errors='ignore')
        else:
            logger.warning("Coluna 'anomaly' não encontrada no SKAB")
            y = pd.Series()
            X = df
        
        logger.info(f"SKAB preprocessado: {X.shape[1]} features, {sum(y)} anomalias")
        
        return X, y
    
    def get_info(self) -> Dict:
        """Retorna informações sobre o dataset SKAB."""
        return {
            'name': 'Skoltech Anomaly Benchmark (SKAB)',
            'source': 'Kaggle / Skoltech',
            'url': 'https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab',
            'description': 'Water circulation system with labeled anomalies',
            'has_labels': True,
            'label_type': 'binary',
            'temporal': True,
            'features': 'Various sensors (pressure, temperature, vibration, etc.)',
            'note': 'Multiple CSV files, each representing different anomaly scenarios'
        }


class NABLoader:
    """
    Loader para Numenta Anomaly Benchmark (NAB).
    Coleção de séries temporais reais e artificiais com anomalias anotadas.
    
    Dataset: https://github.com/numenta/NAB
    Features: Valores de séries temporais univariadas
    Labels: Timestamps de anomalias
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: Caminho para a pasta do NAB. Se None, usa o caminho padrão.
        """
        if data_path is None:
            data_path = "data/raw/nab/"
        self.data_path = Path(data_path)
        
    def load_category(self, category: str = "realKnownCause") -> List[Dict]:
        """
        Carrega todos os arquivos de uma categoria do NAB.
        
        Categorias disponíveis:
        - realKnownCause: Dados reais com causa conhecida
        - realAdExchange: Dados de anúncios online
        - realTraffic: Dados de tráfego
        - artificialWithAnomaly: Dados sintéticos com anomalias
        - artificialNoAnomaly: Dados sintéticos sem anomalias
        
        Args:
            category: Nome da categoria
            
        Returns:
            Lista de dicts com 'name' e 'data' (DataFrame)
        """
        category_path = self.data_path / category
        
        if not category_path.exists():
            logger.warning(f"Categoria NAB não encontrada: {category_path}")
            logger.info("Categorias disponíveis:")
            if self.data_path.exists():
                for d in self.data_path.iterdir():
                    if d.is_dir():
                        logger.info(f"  - {d.name}")
            return []
        
        datasets = []
        for csv_file in category_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
                datasets.append({
                    'name': csv_file.stem,
                    'category': category,
                    'data': df
                })
                logger.debug(f"NAB carregado: {csv_file.name} - {df.shape}")
            except Exception as e:
                logger.warning(f"Erro ao carregar {csv_file.name}: {e}")
        
        logger.info(f"NAB: {len(datasets)} datasets carregados da categoria '{category}'")
        return datasets
    
    def load_all_categories(self) -> Dict[str, List[Dict]]:
        """
        Carrega todas as categorias disponíveis do NAB.
        
        Returns:
            Dict mapeando categoria -> lista de datasets
        """
        if not self.data_path.exists():
            logger.error(f"Diretório NAB não encontrado: {self.data_path}")
            return {}
        
        all_data = {}
        categories = [
            'realKnownCause',
            'realAdExchange', 
            'realTraffic',
            'artificialWithAnomaly',
            'artificialNoAnomaly'
        ]
        
        for category in categories:
            datasets = self.load_category(category)
            if datasets:
                all_data[category] = datasets
        
        total = sum(len(datasets) for datasets in all_data.values())
        logger.info(f"NAB: {total} datasets carregados de {len(all_data)} categorias")
        
        return all_data
    
    def load_labels(self, labels_file: str = "labels/combined_windows.json") -> Dict:
        """
        Carrega arquivo de labels com janelas de anomalias.
        
        Args:
            labels_file: Caminho relativo ao diretório NAB
            
        Returns:
            Dict mapeando nome do arquivo -> lista de janelas de anomalias
        """
        import json
        
        labels_path = self.data_path.parent / labels_file
        
        if not labels_path.exists():
            logger.warning(f"Arquivo de labels não encontrado: {labels_path}")
            return {}
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        logger.info(f"NAB labels carregados: {len(labels)} arquivos")
        return labels
    
    def get_info(self) -> Dict:
        """Retorna informações sobre o NAB."""
        return {
            'name': 'Numenta Anomaly Benchmark (NAB)',
            'source': 'Numenta / GitHub',
            'url': 'https://github.com/numenta/NAB',
            'description': 'Real-world and artificial time series with labeled anomalies',
            'has_labels': True,
            'label_type': 'temporal windows',
            'temporal': True,
            'categories': [
                'realKnownCause',
                'realAdExchange',
                'realTraffic',
                'artificialWithAnomaly',
                'artificialNoAnomaly'
            ]
        }


class NewEnergyVehicleFaultLoader:
    """
    Loader para o dataset de diagnóstico de falhas em veículos de energia nova.
    
    Dataset: https://www.kaggle.com/datasets/ziya07/fault-diagnosis-dataset-for-new-energy-vehicles
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: Caminho para o arquivo. Se None, usa o caminho padrão.
        """
        if data_path is None:
            data_path = "data/raw/new_energy_vehicle_faults.csv"
        self.data_path = Path(data_path)
    
    def load(self) -> pd.DataFrame:
        """Carrega o dataset de falhas de veículos elétricos."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.data_path}\n"
                "Baixe de: https://www.kaggle.com/datasets/ziya07/fault-diagnosis-dataset-for-new-energy-vehicles"
            )
        
        df = pd.read_csv(self.data_path)
        logger.info(f"New Energy Vehicle dataset carregado: {df.shape}")
        return df
    
    def get_info(self) -> Dict:
        """Retorna informações sobre o dataset."""
        return {
            'name': 'Fault Diagnosis Dataset for New Energy Vehicles',
            'source': 'Kaggle',
            'url': 'https://www.kaggle.com/datasets/ziya07/fault-diagnosis-dataset-for-new-energy-vehicles',
            'has_labels': True,
            'description': 'Fault diagnosis data from new energy vehicles'
        }


class PowerSystemFaultsLoader:
    """
    Loader para o dataset de falhas em sistemas de energia.
    
    Dataset: https://www.kaggle.com/datasets/ziya07/power-system-faults-dataset
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: Caminho para o arquivo. Se None, usa o caminho padrão.
        """
        if data_path is None:
            data_path = "data/raw/power_system_faults.csv"
        self.data_path = Path(data_path)
    
    def load(self) -> pd.DataFrame:
        """Carrega o dataset de falhas de sistemas de energia."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.data_path}\n"
                "Baixe de: https://www.kaggle.com/datasets/ziya07/power-system-faults-dataset"
            )
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Power System Faults dataset carregado: {df.shape}")
        return df
    
    def get_info(self) -> Dict:
        """Retorna informações sobre o dataset."""
        return {
            'name': 'Power System Faults Dataset',
            'source': 'Kaggle',
            'url': 'https://www.kaggle.com/datasets/ziya07/power-system-faults-dataset',
            'has_labels': True,
            'description': 'Power system fault detection and classification data'
        }


def get_available_source_loaders() -> Dict[str, type]:
    """
    Retorna um dicionário com todos os loaders de datasets fonte disponíveis.
    
    Returns:
        Dict mapeando nome do dataset -> classe loader
    """
    return {
        'electrical_fault': ElectricalFaultLoader,
        'skab': SKABLoader,
        'nab': NABLoader,
        'new_energy_vehicle': NewEnergyVehicleFaultLoader,
        'power_system': PowerSystemFaultsLoader
    }


def list_available_datasets() -> None:
    """
    Lista todos os datasets fonte disponíveis com suas informações.
    Útil para exploração inicial.
    """
    loaders = get_available_source_loaders()
    
    print("=" * 80)
    print("DATASETS FONTE DISPONÍVEIS")
    print("=" * 80)
    
    for name, loader_class in loaders.items():
        print(f"\n📁 {name}")
        print("-" * 80)
        
        try:
            loader = loader_class()
            info = loader.get_info()
            
            print(f"Nome: {info['name']}")
            print(f"Fonte: {info['source']}")
            if 'url' in info:
                print(f"URL: {info['url']}")
            print(f"Labels: {'Sim' if info.get('has_labels') else 'Não'}")
            if 'description' in info:
                print(f"Descrição: {info['description']}")
        except Exception as e:
            print(f"Erro ao obter informações: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Exemplo de uso e teste dos loaders
    list_available_datasets()