"""Treino do modelo final e serializacao com joblib.

Uso:
    python -m src.treino                      # grava models/pipeline_modelo.joblib
    python -m src.treino --submissao saida.csv # tambem gera previsoes da base de teste
"""
import argparse
import json
import platform
from pathlib import Path

import joblib
import lightgbm
import pandas as pd
import sklearn

from src.config import COL_ALVO, COL_ID, COL_SAFRA, FEATURES, MODELS_DIR
from src.data import carregar_bases, montar_desenvolvimento, montar_teste
from src.features import construir_pipeline, preparar_entrada

NOME_MODELO = "pipeline_modelo.joblib"
NOME_METADADOS = "pipeline_modelo.json"


def treinar(df):
    """Ajusta o Pipeline completo (imputacao, codificacao, LightGBM) em todo o desenvolvimento."""
    X = preparar_entrada(df, FEATURES)
    return construir_pipeline(FEATURES).fit(X, df[COL_ALVO])


def salvar(pipeline, pasta=None, n_linhas=None):
    pasta = Path(pasta) if pasta else MODELS_DIR
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / NOME_MODELO
    joblib.dump(pipeline, caminho)
    metadados = {
        "features": FEATURES,
        "n_linhas_treino": n_linhas,
        "versoes": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    (pasta / NOME_METADADOS).write_text(json.dumps(metadados, indent=2), encoding="utf-8")
    return caminho


def carregar(caminho=None):
    return joblib.load(Path(caminho) if caminho else MODELS_DIR / NOME_MODELO)


def prever(pipeline, df):
    return pipeline.predict_proba(preparar_entrada(df, FEATURES))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissao", help="caminho do CSV de previsoes da base de teste")
    args = ap.parse_args()
    bases = carregar_bases()
    dev = montar_desenvolvimento(bases)
    pipeline = treinar(dev)
    print("Modelo salvo em", salvar(pipeline, n_linhas=len(dev)))
    if args.submissao:
        teste = montar_teste(bases)
        saida = teste[[COL_ID, COL_SAFRA]].copy()
        saida["PROBABILIDADE_INADIMPLENCIA"] = prever(pipeline, teste)
        saida.to_csv(args.submissao, index=False)
        print("Previsoes gravadas em", args.submissao, f"({len(saida)} linhas)")


if __name__ == "__main__":
    main()
