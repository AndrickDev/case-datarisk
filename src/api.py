"""API de escore. Uso: MODELO_PATH=models/pipeline_modelo.joblib uvicorn src.api:app"""
import os
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FEATURES, MODELS_DIR
from src.data import derivar_variaveis
from src.treino import NOME_MODELO, carregar, prever


class Registro(BaseModel):
    """Um documento a ser pontuado. Campos cadastrais ausentes sao tratados como nulos."""

    DATA_EMISSAO_DOCUMENTO: str = Field(..., examples=["2021-07-14"])
    DATA_VENCIMENTO: str = Field(..., examples=["2021-08-04"])
    VALOR_A_PAGAR: Optional[float] = None
    TAXA: Optional[float] = None
    DATA_CADASTRO: Optional[str] = None
    RENDA_MES_ANTERIOR: Optional[float] = None
    NO_FUNCIONARIOS: Optional[float] = None
    DDD: Optional[float] = None
    SEGMENTO_INDUSTRIAL: Optional[str] = None
    DOMINIO_EMAIL: Optional[str] = None
    PORTE: Optional[str] = None
    CEP_2_DIG: Optional[float] = None


class Requisicao(BaseModel):
    registros: List[Registro] = Field(..., min_length=1, max_length=10000)


class Resposta(BaseModel):
    probabilidades: List[float]
    n_registros: int


def criar_app(caminho_modelo=None):
    app = FastAPI(title="Escore de inadimplencia", version="1.0.0")
    app.state.caminho_modelo = caminho_modelo or os.environ.get(
        "MODELO_PATH", str(MODELS_DIR / NOME_MODELO)
    )
    app.state.modelo = None

    def modelo():
        if app.state.modelo is None:
            try:
                app.state.modelo = carregar(app.state.caminho_modelo)
            except FileNotFoundError as erro:
                raise HTTPException(status_code=503, detail="Modelo nao encontrado") from erro
        return app.state.modelo

    @app.get("/saude")
    def saude():
        return {"status": "ok", "features": FEATURES}

    @app.post("/score", response_model=Resposta)
    def score(req: Requisicao):
        df = pd.DataFrame([r.model_dump() for r in req.registros])
        df = derivar_variaveis(df)
        prob = prever(modelo(), df)
        return Resposta(probabilidades=[float(p) for p in prob], n_registros=len(prob))

    return app


app = criar_app()
