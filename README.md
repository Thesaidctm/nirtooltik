# nircls — NIR Species Classification Toolkit

Toolkit em Python para classificação de espécies a partir de espectroscopia no
infravermelho próximo (NIR). O pacote oferece uma interface de linha de comando
e utilidades para criação de pipelines de pré‑processamento, seleção de modelos
e geração de relatórios completos de avaliação.

## Sumário

1. [Instalação e requisitos](#instalacao-e-requisitos)
2. [Configuração via YAML](#configuracao-via-yaml)
3. [Interface de linha de comando](#interface-de-linha-de-comando)
4. [Resultados gerados](#resultados-gerados)
5. [Métricas de calibração e validação](#metricas-de-calibracao-e-validacao)
6. [Pré-processadores e modelos suportados](#pre-processadores-e-modelos-suportados)
7. [Treinar modelo final e predições](#treinar-modelo-final-e-predicoes)
8. [Extensões e contribuições](#extensoes-e-contribuicoes)

## Instalação e requisitos

Requer Python 3.9 ou superior. Para utilizar o toolkit em modo de desenvolvimento
instale no ambiente atual:

```bash
pip install -e .
```

Principais dependências: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`,
`seaborn`, `pyyaml`, `rich`, `joblib`, `openpyxl` e `tqdm`.

## Configuração via YAML

Os experimentos são definidos em arquivos YAML. Exemplo simplificado:

```yaml
data:
  input_path: dados/espectros.xlsx
  target_col: especie

preprocess:
  steps: [SNV, SG]
  sg:
    window: [5, 7]
    polyorder: 2
    deriv: 1

models:
  - name: PLSDA
    grid:
      n_components: [2, 4, 6]

cv:
  random_state: 0
  strategies:
    - type: kfold
      n_splits: [5]
```

* `preprocess.steps` lista os transformadores aplicados antes do classificador.
  Alguns pré-processos como `SG` aceitam parâmetros (janela, ordem do polinômio,
  derivada) que podem ser fixos ou listas para busca em grade.
* `models` define um ou mais estimadores; cada um pode receber um dicionário de
  hiperparâmetros (`grid`).
* `cv.strategies` especifica a estratégia de avaliação: `kfold`, `loo` ou
  `holdout`. É possível combinar múltiplas estratégias.

## Interface de linha de comando

O comando principal é `nircls` e suporta várias sub‑rotinas:

```bash
# Validação cruzada/k-fold
nircls crossval --config experimento.yml

# Holdout simples (definido na seção cv.strategies como type: holdout)
nircls crossval --config experimento.yml

# Treino do melhor modelo e salvamento
nircls train --config experimento.yml

# Predições usando um modelo salvo
nircls predict --model caminho/model.joblib --data novos.xlsx --output preds
```

## Resultados gerados

Cada execução cria uma pasta em `results/<run_id>/<estratégia>/` onde
`<run_id>` identifica a data/hora do experimento. Dentro de cada estratégia são
produzidos:

* `metrics.json` – métricas detalhadas por partição;
* `metrics_summary.csv` – resumo das métricas agregadas;
* `report.md` – relatório em Markdown com tabelas e figuras;
* `figures/` – gráficos de curva ROC, matriz de confusão e confiabilidade.

## Métricas de calibração e validação

Os arquivos de métricas contêm resultados para calibração (dados de treino) e
validação/teste. As métricas de treino recebem o prefixo `train_` e podem incluir
`accuracy`, `f1_macro`, `balanced_accuracy`, `brier_score` e o erro de calibração
esperado (`ece`). A comparação entre `train_` e métricas sem prefixo ajuda a
avaliar overfitting e o desempenho em dados não vistos.

## Pré-processadores e modelos suportados

Pré-processadores disponíveis:

* `SNV` – Standard Normal Variate
* `SG` – Filtro Savitzky‑Golay
* `MSC` – Multiplicative Scatter Correction
* `Detrend` – remoção de tendências polinomiais
* `EMSC` – Extended MSC
* `EPO` – External Parameter Orthogonalization
* `SCALE` – padronização com `StandardScaler`

Modelos implementados:

* `PLSDA`
* `SVM_RBF` e `SVM_LINEAR`
* `RF` (Random Forest)
* `GB` (Gradient Boosting)
* `LOGREG` (Regressão Logística)
* `MLP` (rede neural Multi‑Layer Perceptron)
* `DT` (Árvore de decisão)
* `KNN`

## Treinar modelo final e predições

O subcomando `train` é um atalho para a função `train_and_save`, que ajusta o
modelo escolhido e grava os artefatos. Para gerar e salvar o melhor modelo:

```bash
nircls train --config experimento.yml
```

Será criado um diretório com `model.joblib`, `prepared_XY.xlsx` e `model_card.md`.
Para realizar predições com o modelo salvo, utilize a função `predict_with_model`
ou o subcomando `predict`:

```bash
nircls predict --model results/<run_id>/model.joblib \
               --data novos.xlsx \
               --output resultados
```

O arquivo `predictions.csv` incluirá `y_pred` e, se disponível nos dados, a
coluna `y_true`.

## Extensões e contribuições

O toolkit foi projetado para ser extensível. Novos pré‑processadores ou modelos
podem ser adicionados implementando classes compatíveis com `scikit-learn` e
registrando‑as nos módulos apropriados. Contribuições são bem‑vindas via pull
requests.

