"""Pre-processamento e pipeline. Treino e inferencia usam a mesma lista de variaveis."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    COLUNAS_PROIBIDAS,
    FEATURES,
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
    PARAMS_LGBM,
)


def _texto(valor):
    if pd.isna(valor):
        return "NA"
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return str(int(valor)) if float(valor).is_integer() else str(valor)
    return str(valor)


def preparar_entrada(df, features=None):
    """Seleciona a lista unica de variaveis e padroniza tipos.

    Colunas ausentes no DataFrame de entrada viram valores nulos, de modo que a
    inferencia aceita registros sem cadastro completo.
    """
    features = list(features) if features is not None else FEATURES
    proibidas = set(features) & set(COLUNAS_PROIBIDAS)
    if proibidas:
        raise ValueError(f"Variaveis proibidas na lista de features: {sorted(proibidas)}")
    saida = pd.DataFrame(index=df.index)
    for col in features:
        serie = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        if col in FEATURES_CATEGORICAS:
            saida[col] = serie.map(_texto)
        else:
            saida[col] = pd.to_numeric(serie, errors="coerce")
    return saida


def construir_preprocessador(features=None):
    features = list(features) if features is not None else FEATURES
    num = [c for c in features if c in FEATURES_NUMERICAS]
    cat = [c for c in features if c in FEATURES_CATEGORICAS]
    transformadores = []
    if num:
        transformadores.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformadores.append(("cat", codificador, cat))
    prep = ColumnTransformer(transformadores, verbose_feature_names_out=False)
    prep.set_output(transform="pandas")
    return prep


def construir_pipeline(features=None, params=None):
    """Pipeline sklearn: imputacao mediana, codificacao ordinal e LightGBM."""
    return Pipeline([
        ("prep", construir_preprocessador(features)),
        ("modelo", lgb.LGBMClassifier(**(params or PARAMS_LGBM))),
    ])
