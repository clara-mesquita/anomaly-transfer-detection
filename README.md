# Anomaly Transfer Detection - Electrical Systems

## Visão Geral

Este projeto implementa detecção de anomalias em dados elétricos usando uma abordagem de transferência de conhecimento entre datasets. A ideia principal é usar datasets rotulados de falhas elétricas (source) para melhorar a detecção de anomalias em datasets de veículos não rotulados ou fracamente rotulados (target).

## Motivação

Na prática, é muito difícil avaliar detecção de anomalias não supervisionada sem rótulos ou especialistas de domínio. Datasets de veículos geralmente não têm anotações de falhas explícitas. Portanto, investigamos se modelos treinados em datasets elétricos rotulados podem ser transferidos para contextos veiculares e ainda serem úteis.

## Datasets

### Source
- **Electrical Fault Detection and Classification**: Dataset principal com 506 registros de falhas elétricas
- Opcionalmente: NAB, SKAB, New Energy Vehicles, Power System Faults

### Target
- **CAN Bus Anomaly Detection Dataset**: 46,623 registros de dados de veículos
- Opcionalmente: VED, can-train-and-test, Current characteristic train

## Instalação

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Estrutura de Dados

Coloque os datasets na pasta `data/raw/`:
- `data/raw/electrical_fault_detection.csv`
- `data/raw/can_bus_anomaly_detection.csv`

## Uso

### 1. Treinamento no Dataset Fonte

```bash
python scripts/train_source.py \
    --dataset electrical_fault \
    --model isolation_forest \
    --output models_exported/
```

### 2. Experimento de Transferência

```bash
python scripts/transfer_experiment.py \
    --source_model models_exported/isolation_forest_source.pkl \
    --target_dataset can_bus \
    --output results/
```

### 3. API de Serving

```bash
cd src/api
uvicorn main:app --reload --port 8000
```

Endpoints disponíveis:
- `GET /health` - Health check
- `GET /info` - Informações sobre o modelo
- `POST /predict` - Predição de anomalias

Exemplo de requisição:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "Voltage": 230.5,
        "Current": 15.2,
        "Temperature": 45.3,
        "Pressure": 101.3
      }
    ]
  }'
```

## Notebooks

Execute os notebooks em ordem:
1. `01_eda_and_transfer_experiments.ipynb` - EDA completa e experimentos de transferência

## Reprodutibilidade

Todos os processos aleatórios usam seeds fixas (42) para garantir reprodutibilidade.

## Estrutura do Código

- `src/data_loading/` - Carregamento de datasets
- `src/preprocessing/` - Normalização, codificação, alinhamento de features
- `src/models/` - Implementação de modelos (Isolation Forest, KMeans)
- `src/transfer/` - Lógica de transferência entre domínios
- `src/api/` - API FastAPI para serving de modelos


