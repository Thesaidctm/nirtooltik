# nircls — NIR Species Classification Toolkit
Autor: Thalles Loiola Dias, Discente da UFLA

Toolkit em Python para classificação de espécies a partir de espectroscopia no
infravermelho próximo (NIR). O pacote oferece uma interface de linha de comando
e utilidades para criação de pipelines de pré‑processamento, seleção de modelos
e geração de relatórios completos de avaliação.

## Sumário

1. [Instalação e requisitos](#instalacao-e-requisitos)
2. [Configuração via YAML](#configuracao-via-yaml)
3. [Interface de linha de comando](#interface-de-linha-de-comando)
4. [Dashboard interativo](#dashboard-interativo)
5. [Resultados gerados](#resultados-gerados)
6. [Métricas de calibração e validação](#metricas-de-calibracao-e-validacao)
7. [Pré-processadores e modelos suportados](#pre-processadores-e-modelos-suportados)
8. [Treinar modelo final e predições](#treinar-modelo-final-e-predicoes)
9. [Erros comuns e soluções](#erros-comuns-e-solucoes)
10. [Extensões e contribuições](#extensoes-e-contribuicoes)
11. [Testes](#testes)

## Instalação e requisitos

Requer Python 3.9 ou superior. Para utilizar o toolkit em modo de desenvolvimento
instale no ambiente atual:

```bash
pip install -e .
```

Principais dependências: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`,
`seaborn`, `pyyaml`, `rich`, `joblib`, `openpyxl` e `tqdm`.

Para exportar relatórios em PDF é necessário instalar dependências extras:

* `markdown` e `weasyprint` (recomendado), ou
* `markdown` e `pdfkit`, juntamente com o executável `wkhtmltopdf`.

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
# Assistente interativo para gerar config.yaml
nircls wizard

# Execução de demonstração com dados de exemplo
nircls demo

# Validação cruzada/k-fold
nircls crossval --config experimento.yml

# Holdout simples (definido na seção cv.strategies como type: holdout)
nircls crossval --config experimento.yml

# Treino do melhor modelo e salvamento
nircls train --config experimento.yml

# Predições usando um modelo salvo
nircls predict --model caminho/model.joblib --data novos.xlsx --output preds

# Dashboard interativo
nircls dashboard --results results
```

A maioria dos subcomandos aceita o argumento global `--seed` para definir a
semente das operações aleatórias (como divisão de dados ou embaralhamento),
permitindo reproduzir resultados. Por padrão utiliza-se o valor `42`.

```bash
# Exemplo com semente fixa
nircls train --config experimento.yml --seed 123
```

O subcomando `wizard` abre um assistente que solicita os caminhos dos dados,
coluna alvo, modelos e parâmetros básicos, gerando automaticamente um
`config.yaml`. Ao final é possível iniciar imediatamente o treinamento e a
avaliação com base nas escolhas feitas.

Consulte a seção [Dashboard interativo](#dashboard-interativo) para executar
experimentos completos pela interface web.

## Dashboard interativo

O dashboard permite executar todas as etapas do fluxo sem recorrer à CLI.

1. Inicie a aplicação:

   ```bash
   nircls dashboard --results results
   ```

   A pasta indicada armazenará os artefatos gerados.

2. Na aba **Preparar Dados**, envie o arquivo CSV/XLSX, escolha a coluna alvo e
   opcionalmente informe o intervalo espectral. Ao enviar o formulário a função
   `prepare_inputs` processa os dados.

3. Em **Treinar**, selecione o modelo desejado e forneça um grid de parâmetros
   em JSON. O botão *Treinar* chama `train_and_save` e grava o modelo treinado na
   pasta de resultados.

4. Em **Avaliar**, configure o número de folds, modelo e parâmetros. O formulário
   executa `run_multiple_eval` gerando métricas e relatórios. A seção também
   permite enviar um modelo salvo e um novo arquivo de dados para que
   `predict_with_model` produza o arquivo `predictions.csv`.

Com essas etapas é possível conduzir um experimento completo apenas pela
interface web.

## Resultados gerados

Cada execução cria uma pasta em `results/<run_id>/<estratégia>/` onde
`<run_id>` identifica a data/hora do experimento. Dentro de cada estratégia são
produzidos:

* `metrics.json` – métricas detalhadas por partição;
* `metrics_summary.csv` – resumo das métricas agregadas;
* `report.md` – relatório em Markdown com tabelas e figuras;
* `figures/` – gráficos de curva ROC, matriz de confusão e confiabilidade.

O relatório é gerado inteiramente de forma local, sem qualquer integração com
serviços externos ou APIs.

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

* `PLSDA` – usa regressão de mínimos quadrados parciais para discriminação,
  adequada a dados de alta dimensionalidade com colinearidade
* `SVM_RBF` e `SVM_LINEAR` – máquinas de vetores de suporte; kernel RBF modela
  fronteiras não lineares enquanto o kernel linear serve para separações simples
* `RF` (Random Forest) – conjunto de árvores de decisão construídas por bootstrap
  que reduz o overfitting
* `GB` (Gradient Boosting) – adiciona modelos fracos sequencialmente para
  corrigir erros residuais e capturar padrões complexos
* `LOGREG` (Regressão Logística) – modelo linear probabilístico para
  classificação binária ou multiclasse
* `MLP` (rede neural Multi‑Layer Perceptron) – camadas densas capazes de
  aprender relações não lineares
* `DT` (Árvore de decisão) – estrutura hierárquica que particiona os dados por
  atributos, fácil de interpretar mas sensível a ruído
* `KNN` – atribui o rótulo mais comum entre os vizinhos mais próximos,
  dispensando fase de treinamento explícita
* `ExtraTrees` (Extremely Randomized Trees) – florestas com divisões totalmente aleatórias,
  indicadas para reduzir variância em dados ruidosos
* `AdaBoost`, `XGBoost`, `LightGBM`, `CatBoost` – métodos de boosting robusto que capturam
  relações complexas e lidam bem com grandes conjuntos de dados
* `QDA` (Quadratic Discriminant Analysis) – adequado quando cada classe tem matrizes de
  covariância distintas
* `Naive Bayes` – assume independência condicional entre atributos, servindo como baseline
  rápido em alta dimensionalidade
* `CNN 1D` e `LSTM 1D` – redes neurais para séries 1D; convoluções extraem padrões locais
  enquanto LSTMs modelam dependências de longo prazo

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

## Erros comuns e soluções

* **Arquivo de configuração não encontrado** – verifique o caminho informado na opção `--config`.
* **Coluna alvo ausente** – confirme se a planilha contém a coluna especificada em `data.target_cols`.
* **Arquivo de dados não encontrado** – confira o valor de `data.input_path` no YAML.
* **Modelo incompatível** – utilize um nome de modelo suportado e revise os parâmetros.

## Extensões e contribuições

O toolkit foi projetado para ser extensível. Novos pré‑processadores ou modelos
podem ser adicionados implementando classes compatíveis com `scikit-learn` e
registrando‑as nos módulos apropriados. Contribuições são bem‑vindas via pull
requests.

Os resultados serão salvos em `results/<run_id>/<estratégia>/`. Cada pasta de
estratégia contém as métricas (`metrics.json`, `metrics_summary.xlsx`), as figuras
geradas e um relatório final em Markdown (`report.md`).

## Testes

Para rodar a suíte de testes utilize o [pytest](https://pytest.org):

```bash
pytest
```


## Créditos/Autoria

Desenvolvido por Thalles Loiola Dias, Discente da UFLA.

