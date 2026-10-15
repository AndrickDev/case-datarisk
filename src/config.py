"""Constantes do projeto: alvo, identificadores e lista unica de variaveis."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", RAIZ / "data"))
REPORTS_DIR = RAIZ / "reports"
MODELS_DIR = RAIZ / "models"

LIMITE_ATRASO_DIAS = 5
COL_ID = "ID_CLIENTE"
COL_SAFRA = "SAFRA_REF"
COL_ALVO = "INADIMPLENTE"
SEMENTE = 42

# Colunas que identificam cliente ou derivam do desfecho. Nunca entram no modelo.
COLUNAS_PROIBIDAS = [
    COL_ID,
    COL_SAFRA,
    COL_ALVO,
    "DIAS_ATRASO",
    "DATA_PAGAMENTO",
]

# Lista unica de variaveis, usada no treino, na validacao e na inferencia.
FEATURES_NUMERICAS = [
    "VALOR_A_PAGAR",
    "TAXA",
    "RENDA_MES_ANTERIOR",
    "NO_FUNCIONARIOS",
    "MES_EMISSAO",
    "DIA_EMISSAO",
    "DIA_VENCIMENTO",
    "PRAZO_DIAS",
    "TEMPO_CADASTRO_DIAS",
]
FEATURES_CATEGORICAS = [
    "DDD",
    "SEGMENTO_INDUSTRIAL",
    "DOMINIO_EMAIL",
    "PORTE",
    "CEP_2_DIG",
]
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

# Hiperparametros do LightGBM (melhor combinacao do GridSearchCV do notebook).
PARAMS_LGBM = dict(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=40,
    random_state=SEMENTE,
    verbose=-1,
)
