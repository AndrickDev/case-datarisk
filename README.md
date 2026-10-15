# Probabilidade de inadimplência – Case Datarisk (Cientista de Dados Júnior 2025)

Modelo de probabilidade de inadimplência (PD) para cobranças de clientes pessoa jurídica. O alvo `INADIMPLENTE` é 1 quando o pagamento ocorre com 5 ou mais dias de atraso em relação ao vencimento (taxa observada de 7,02% em 77.414 documentos e 1.248 clientes).

O repositório contém o notebook original do case, mantido como registro, e um pacote em `src/` com a solução revisada: lista única de variáveis, validação por grupo de cliente e out-of-time, métricas de decisão, calibração, monitoramento de drift, explicabilidade, serialização do modelo e API de escore.

Todos os números deste documento foram gerados pelos scripts de `src/` e estão em `reports/`. Os dados do desafio não são versionados.

## Resultados

AUC, KS e Gini do LightGBM com hiperparâmetros do notebook (300 árvores, profundidade 10, 40 folhas, taxa de aprendizado 0,1), conjunto final de 14 variáveis. Validação aleatória e por grupo usam 5 folds (média dos folds); out-of-time treina nas safras anteriores e testa nas k mais recentes.

| Esquema de validação | AUC | KS | Gini |
|---|---|---|---|
| Aleatória por linha (StratifiedKFold) | 0,9619 | 0,8160 | 0,9238 |
| Por grupo de cliente (GroupKFold por `ID_CLIENTE`) | 0,7981 | 0,4664 | 0,5961 |
| Out-of-time, 3 safras mais recentes (2021-04 a 2021-06) | 0,9017 | 0,7104 | 0,8034 |
| Out-of-time, 6 safras mais recentes (2021-01 a 2021-06) | 0,9022 | 0,6763 | 0,8045 |

Leitura: a validação aleatória é a mais otimista porque documentos do mesmo cliente caem em treino e validação. A validação por grupo estima o desempenho em clientes não vistos (AUC 0,80). A validação out-of-time estima o desempenho em safras futuras para clientes já conhecidos (AUC 0,90), que é o cenário da base de teste do desafio: 98,1% dos documentos da base de teste pertencem a clientes presentes no desenvolvimento.

### Comparação entre conjuntos de variáveis

| Conjunto | Aleatória (AUC) | Grupo (AUC) | Out-of-time k=6 (AUC) |
|---|---|---|---|
| Notebook sem `ID_CLIENTE` (5 variáveis) | 0,8240 | 0,7465 | 0,7018 |
| Completo (14 variáveis, adotado) | 0,9619 | 0,7981 | 0,9022 |
| Completo sem `TAXA` e `VALOR_A_PAGAR` | 0,9511 | 0,7553 | 0,8764 |

KS e Gini de cada conjunto estão em `reports/validacao.json`.

### Auditoria de vazamento do modelo original

O LightGBM final do notebook usa `X.columns = ['ID_CLIENTE', 'VALOR_A_PAGAR', 'TAXA', 'MES_EMISSAO', 'DIA_EMISSAO', 'DIA_VENCIMENTO']`. O identificador entra porque `X` é montado com `select_dtypes` sobre colunas numéricas e `ID_CLIENTE` é `int64`. As variáveis cadastrais e de renda não chegam a esse modelo.

| Cenário (modelo do notebook) | Aleatória | Por grupo de cliente | Out-of-time k=6 |
|---|---|---|---|
| Com `ID_CLIENTE` | 0,9299 | 0,6999 | 0,8256 |
| Sem `ID_CLIENTE` | 0,8227 | 0,7473 | 0,7022 |

O AUC de 0,9299 do notebook inclui o identificador. Sem ele, cai para 0,8227 na mesma validação. O identificador piora o desempenho por grupo de cliente (0,6999 contra 0,7473) e o melhora out-of-time (0,8256 contra 0,7022), o que é compatível com um efeito fixo de cliente: ajuda quando o cliente já foi visto e não generaliza para clientes novos. O valor de 0,8227 usa `StratifiedKFold(5, shuffle=True, random_state=42)` sobre as colunas do notebook; a diferença para os 0,8240 da tabela anterior vem do pré-processamento do pipeline do pacote (imputação pela mediana dentro do pipeline).

### Decisão sobre as variáveis

`PORTE`, `RENDA_MES_ANTERIOR` e `SEGMENTO_INDUSTRIAL` entram no modelo final, junto com as demais variáveis cadastrais, de renda e de prazo (lista única em `src/config.py`). Critério: o conjunto completo supera o conjunto reduzido em todos os esquemas, com AUC out-of-time de 0,9022 contra 0,7018 e por grupo de 0,7981 contra 0,7465. `FLAG_PF` foi excluída por ser nula em todos os registros de desenvolvimento. `ID_CLIENTE`, `SAFRA_REF` e colunas derivadas do pagamento são rejeitadas por `preparar_entrada`.

### Calibração e limiar de decisão

Protocolo (`src/decisao.py`): modelo treinado até 2020-09; calibração e escolha de limiar em 2020-10 a 2020-12 (7.205 documentos); teste out-of-time em 2021-01 a 2021-06 (14.267 documentos, taxa de 6,29%).

| Probabilidade | Brier | Média prevista (observada: 0,0629) |
|---|---|---|
| Bruta (LightGBM) | 0,0431 | 0,0543 |
| Platt | 0,0433 | 0,0744 |
| Isotônica | 0,0438 | 0,0751 |

Nenhuma das duas calibrações melhorou o Brier no teste. A janela de calibração teve taxa de inadimplência diferente da do teste, o que deslocou a média prevista dos modelos calibrados para cima. A curva de confiabilidade está em `reports/calibracao.png`.

Matriz de confusão no teste, probabilidade isotônica, limiar escolhido na janela de calibração minimizando `custo_FN * FN + custo_FP * FP`. As razões de custo são hipóteses de sensibilidade, não dados do desafio.

| Custo FN / custo FP | Limiar | Precisão | Recall | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| 1 | 0,5000 | 0,599 | 0,408 | 366 | 245 | 531 | 13.125 |
| 2 | 0,3333 | 0,554 | 0,439 | 394 | 317 | 503 | 13.053 |
| 5 | 0,2094 | 0,406 | 0,589 | 528 | 772 | 369 | 12.598 |
| 10 | 0,1003 | 0,174 | 0,790 | 709 | 3.362 | 188 | 10.008 |

Com limiar fixo de 0,5 e probabilidade bruta: precisão 0,665 e recall 0,321 (`reports/decisao_calibracao.json`).

### Drift (PSI e KS por safra)

Referência das variáveis: safras de 2018-08 a 2020-12. Referência do escore: 2020-10 a 2020-12, em predição fora da amostra de treino. Monitoradas: 2021-01 a 2021-06 (teste out-of-time) e 2021-07 a 2021-11 (base de teste do desafio). Alerta para PSI > 0,25 (`reports/drift.json`).

- Escore: PSI de 0,502 em 2021-02 e 0,603 em 2021-03; nas demais safras, entre 0,011 e 0,072. Nas safras da base de teste (2021-07 a 2021-11) o PSI do escore fica entre 0,017 e 0,072. O KS do escore contra a referência é 0,314 e 0,338 nas duas safras em alerta e no máximo 0,110 nas demais.
- `VALOR_A_PAGAR`: PSI acima de 0,25 de 2021-04 em diante, de 0,255 a 0,788.
- `NO_FUNCIONARIOS`: PSI entre 0,267 e 0,359 em todas as 11 safras monitoradas.
- `MES_EMISSAO`, `DIA_EMISSAO`, `DIA_VENCIMENTO` e `TEMPO_CADASTRO_DIAS` também excedem 0,25, mas por construção (cada safra concentra um mês e o tempo de cadastro cresce com o calendário). São registradas como alertas estruturais e separadas dos alertas de drift.

### Explicabilidade

SHAP médio absoluto (LightGBM, 5.000 documentos do teste out-of-time; `reports/shap_resumo.png`): `PRAZO_DIAS` 0,68, `VALOR_A_PAGAR` 0,51, `CEP_2_DIG` 0,40, `MES_EMISSAO` 0,39, `TEMPO_CADASTRO_DIAS` 0,26.

IV e odds ratio em regressão logística sobre variáveis em WOE (ajustados até 2020-12; `reports/explicabilidade.json`). WOE definido como ln(%inadimplentes / %adimplentes); o odds ratio é `exp(coeficiente)` por unidade de WOE.

| Variável | IV | Classe | Odds ratio |
|---|---|---|---|
| `PRAZO_DIAS` | 0,893 | forte | 1,73 |
| `CEP_2_DIG` | 0,853 | forte | 1,73 |
| `VALOR_A_PAGAR` | 0,806 | forte | 2,02 |
| `DDD` | 0,785 | forte | 1,56 |
| `RENDA_MES_ANTERIOR` | 0,123 | média | 2,10 |
| `PORTE` | 0,115 | média | 1,66 |
| `TAXA` | 0,011 | não preditiva | 2,00 |

A regressão logística em WOE obteve AUC 0,8472, KS 0,5445 e Gini 0,6944 no teste out-of-time (6 safras), contra 0,9022 do LightGBM. O IV de `CEP_2_DIG` e `DDD` é calculado no mesmo período em que as faixas são ajustadas e essas variáveis têm dezenas de categorias, portanto está inflado.

## Estrutura

```
src/
  config.py            alvo, identificadores proibidos, lista única de variáveis, hiperparâmetros
  data.py              leitura das bases, alvo, variáveis derivadas
  features.py          preparar_entrada e Pipeline sklearn (imputação, codificação, LightGBM)
  metrics.py           AUC, KS, Gini
  auditoria.py         reprodução do vazamento de ID_CLIENTE
  validacao.py         validação aleatória, por grupo e out-of-time
  decisao.py           precisão, recall, matriz de confusão, limiar por custo, calibração
  drift.py             PSI por variável e por escore, KS entre safras
  explicabilidade.py   SHAP, IV, odds ratio
  treino.py            treino final e serialização com joblib
  api.py               API FastAPI
tests/                 30 testes (pytest), com dados sintéticos
reports/               saídas das execuções (JSON e PNG)
Case_DataRisk_Notebook_Final_COMPLETO_APRESENTACAO.ipynb   notebook original
```

## Como executar

Python 3.12. Os arquivos `base_cadastral.csv`, `base_info.csv`, `base_pagamentos_desenvolvimento.csv` e `base_pagamentos_teste.csv` ficam fora do git; aponte a pasta com `DATA_DIR`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
export DATA_DIR=/caminho/para/os/csv

python -m pytest -q
python -m src.auditoria
python -m src.validacao
python -m src.decisao
python -m src.drift
python -m src.explicabilidade
python -m src.treino --submissao previsoes.csv   # grava models/pipeline_modelo.joblib
```

As versões em `requirements.txt` foram instaladas em ambiente limpo, com os 30 testes passando, os scripts acima executados e o notebook executado de ponta a ponta. O modelo serializado só é garantido com a mesma versão de scikit-learn e LightGBM (`models/pipeline_modelo.json` registra as versões usadas no treino).

### API de escore

```bash
MODELO_PATH=models/pipeline_modelo.joblib uvicorn src.api:app
```

`GET /saude` retorna a lista de variáveis. `POST /score` recebe `{"registros": [...]}` com `DATA_EMISSAO_DOCUMENTO` e `DATA_VENCIMENTO` obrigatórios e os demais campos opcionais (campos ausentes são tratados como nulos), e retorna `probabilidades`. Sem modelo no caminho configurado, responde 503.

## Limitações conhecidas

- O desempenho out-of-time (AUC 0,90) vale para clientes já presentes no desenvolvimento. Para clientes novos, a estimativa é a da validação por grupo (AUC 0,80).
- A diferença entre a validação por grupo e a out-of-time sugere que atributos estáticos do cliente (`DDD`, `CEP_2_DIG`, `PORTE`, `TEMPO_CADASTRO_DIAS`, `TAXA`) funcionam como identificadores indiretos. Essa é uma hipótese; não foi testada isoladamente.
- Não há variáveis de comportamento histórico do cliente (atraso em safras anteriores). O modelo não captura mudança de comportamento ao longo do tempo.
- Os hiperparâmetros vêm do GridSearchCV do notebook, feito com validação aleatória, e não foram reotimizados sob validação por grupo ou out-of-time.
- A validação out-of-time usa um único corte (k=3 e k=6). Não há avaliação em janela móvel.
- As razões de custo de falso negativo e falso positivo são hipóteses de sensibilidade. O limiar de produção depende dos custos reais.
- A calibração (Platt e isotônica) não melhorou o Brier, e a janela de calibração tem taxa de inadimplência diferente da do teste.
- Há drift em `VALOR_A_PAGAR` e `NO_FUNCIONARIOS` e deslocamento do escore em 2021-02 e 2021-03.
- A base de desenvolvimento só tem pessoa jurídica; o modelo não se aplica a pessoa física.
- `VALOR_A_PAGAR` tem 1,5% de nulos (1.170 de 77.414), imputados pela mediana.
- O notebook original continua com o vazamento de `ID_CLIENTE` no código (documentado na seção de auditoria do próprio notebook). O arquivo de submissão por ele gerado usa esse modelo; as previsões do modelo revisado são geradas por `python -m src.treino --submissao`.
- O pickle original foi removido do índice do git, mas permanece no histórico de commits anteriores.

## Plano de evolução

1. Construir variáveis de comportamento com defasagem temporal por cliente (taxa de atraso e atraso médio nas safras anteriores), respeitando a data de corte de cada documento.
2. Reotimizar hiperparâmetros com validação por grupo e out-of-time, e avaliar em janela móvel de safras.
3. Testar a hipótese de identificação indireta do cliente, comparando modelos com e sem atributos estáticos sob validação por grupo.
4. Definir limiar com custos reais de falso negativo e falso positivo e reavaliar a calibração quando houver mais safras.
5. Automatizar o monitoramento de PSI e KS por safra com alerta acima de 0,25 e retreino condicionado.
6. Empacotar a API em contêiner e adicionar integração contínua com os testes.
