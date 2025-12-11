"""
Exemplo de uso rápido do projeto de detecção de anomalias com transferência.

Este script demonstra:
1. Como carregar datasets
2. Como treinar um modelo no fonte
3. Como aplicar no alvo
4. Como fazer predições
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data_loading.source_loaders import ElectricalFaultLoader
from src.data_loading.target_loaders import CANBusLoader
from src.preprocessing.scalers import TransferScaler
from src.models.isolation_forest import IsolationForestDetector


def main():
    print("=" * 80)
    print("EXEMPLO DE USO - DETECÇÃO DE ANOMALIAS COM TRANSFERÊNCIA")
    print("=" * 80)
    
    # 1. Carregar e preparar dados fonte
    print("\n[1] Carregando dataset fonte (Electrical Fault Detection)...")
    source_loader = ElectricalFaultLoader()
    
    try:
        X_source, y_source = source_loader.load_preprocessed()
        print(f"✓ Fonte carregado: {X_source.shape[0]} amostras")
    except FileNotFoundError:
        print("✗ Arquivo não encontrado. Por favor, baixe o dataset do Kaggle.")
        print("  URL: https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification")
        return
    
    # Selecionar apenas features que existem no alvo
    common_features_source = ['Voltage (V)', 'Current (A)', 'Temperature (°C)']
    X_source = X_source[common_features_source]
    
    # 2. Treinar modelo no fonte
    print("\n[2] Treinando modelo Isolation Forest no dataset fonte...")
    
    # Normalizar
    scaler = TransferScaler(method='standard')
    X_source_scaled = scaler.fit_transform(X_source)
    
    # Treinar
    model = IsolationForestDetector(
        n_estimators=100,
        contamination=0.3,
        random_state=42
    )
    model.fit(X_source_scaled)
    
    print("✓ Modelo treinado")
    print(f"  Parâmetros: {model.get_params()}")
    
    # 3. Carregar dados alvo
    print("\n[3] Carregando dataset alvo (CAN Bus)...")
    target_loader = CANBusLoader()
    
    try:
        target_df = target_loader.load_preprocessed()
        print(f"✓ Alvo carregado: {target_df.shape[0]} amostras")
    except FileNotFoundError:
        print("✗ Arquivo não encontrado. Por favor, baixe o dataset do Kaggle.")
        print("  URL: https://www.kaggle.com/datasets/ankitrajsh/can-bus-anomaly-detection-dataset")
        return
    
    # Preparar features do alvo
    common_features_target = ['Voltage', 'Current', 'Temperature']
    X_target = target_df[common_features_target].dropna()
    
    # Renomear para match com fonte
    X_target.columns = common_features_source
    
    print(f"  Features alinhadas: {common_features_target}")
    
    # 4. Aplicar modelo treinado no fonte ao alvo
    print("\n[4] Aplicando modelo ao dataset alvo...")
    
    # Usar MESMO scaler treinado no fonte
    X_target_scaled = scaler.transform(X_target)
    
    # Predição
    predictions = model.predict(X_target_scaled)
    scores = model.score_samples(X_target_scaled)
    
    n_anomalies = np.sum(predictions)
    anomaly_rate = np.mean(predictions) * 100
    
    print(f"✓ Predição concluída")
    print(f"  Anomalias detectadas: {n_anomalies} ({anomaly_rate:.2f}%)")
    print(f"  Score médio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 5. Analisar top anomalias
    print("\n[5] Top 5 anomalias mais extremas:")
    top_indices = np.argsort(scores)[-5:][::-1]
    
    print("\nÍndice | Voltage | Current | Temperature | Anomaly Score")
    print("-" * 65)
    for idx in top_indices:
        row = X_target.iloc[idx]
        score = scores[idx]
        print(f"{idx:6d} | {row['Voltage (V)']:7.2f} | {row['Current (A)']:7.2f} | "
              f"{row['Temperature (°C)']:11.2f} | {score:13.4f}")
    
    # 6. Salvar modelo (opcional)
    print("\n[6] Salvando modelo e preprocessador...")
    output_dir = Path("models_exported")
    output_dir.mkdir(exist_ok=True)
    
    model.save(output_dir / "example_model.pkl")
    scaler.save(output_dir / "example_scaler.pkl")
    
    print(f"✓ Modelo salvo em: {output_dir / 'example_model.pkl'}")
    print(f"✓ Scaler salvo em: {output_dir / 'example_scaler.pkl'}")
    
    # 7. Exemplo de como carregar e usar modelo salvo
    print("\n[7] Exemplo de uso do modelo salvo:")
    print("\n# Carregar modelo")
    print("loaded_model = IsolationForestDetector.load('models_exported/example_model.pkl')")
    print("loaded_scaler = TransferScaler.load('models_exported/example_scaler.pkl')")
    print("\n# Novos dados")
    print("new_data = pd.DataFrame({")
    print("    'Voltage (V)': [230.5],")
    print("    'Current (A)': [15.2],")
    print("    'Temperature (°C)': [45.3]")
    print("})")
    print("\n# Predição")
    print("new_data_scaled = loaded_scaler.transform(new_data)")
    print("prediction = loaded_model.predict(new_data_scaled)")
    print("score = loaded_model.score_samples(new_data_scaled)")
    print("\nprint(f'Anomalia: {prediction[0]}')  # 0=normal, 1=anomalia")
    print("print(f'Score: {score[0]:.4f}')")
    
    print("\n" + "=" * 80)
    print("EXEMPLO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print("\nPróximos passos:")
    print("1. Execute o notebook: notebooks/01_eda_and_transfer_experiments.ipynb")
    print("2. Treine modelos: python scripts/train_source.py")
    print("3. Execute experimentos: python scripts/transfer_experiment.py")
    print("4. Inicie a API: cd src/api && uvicorn main:app --reload")


if __name__ == "__main__":
    main()