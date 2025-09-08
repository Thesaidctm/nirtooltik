# NIRCLS — NIR Spectroscopy Toolkit for Tree Species Classification
Autor: Thalles Loiola Dias | PPGCTM | UFLA |

## Introdução

Toolkit em Python para classificação de espécies a partir de espectroscopia no
infravermelho próximo (NIR). O pacote oferece uma interface de linha de comando
e utilidades para criação de pipelines de pré‑processamento, seleção de modelos
e geração de relatórios completos de avaliação. A partir das métricas geradas,
o comando produz automaticamente um relatório científico em Markdown com tabelas
e figuras embutidas, além de metadados de citação. Consulte o notebook
[examples/quickstart.ipynb](examples/quickstart.ipynb) para um exemplo rápido
de uso da API.

A documentação completa está disponível em [https://Thesaidctm.github.io/nirtooltik/](https://Thesaidctm.github.io/nirtooltik/).

Antes de executar os comandos a seguir, recomenda-se criar e ativar um
ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
# No Windows use:
.venv\Scripts\activate
```

Com o ambiente virtual ativo, usuários iniciantes podem executar um pipeline
completo a partir de um arquivo YAML de configuração, que define os caminhos
dos dados e demais opções do experimento. Exemplo:

```bash
nircls run-all --config configs/eucalipto_spxy.yaml
```

O comando exige um arquivo de configuração YAML que informe os caminhos dos
dados e as opções do experimento.

Para treinar todos os modelos disponíveis (pode exigir dependências opcionais), utilize:

```bash
nircls run-all --config configs/eucalipto_spxy.yaml --all-models
```

Para aplicar todos os pré-processamentos suportados (SNV, MSC, DETREND, SG, EMSC, EPO, SCALE) em conjunto com todos os modelos, combine as duas flags:

```bash
nircls run-all --config configs/eucalipto_spxy.yaml --all-models --all-preprocess
```

## Sumário

1. [Introdução](#introducao)
2. [Requisitos](#requisitos)
3. [Instalação](#instalacao)
4. [Configuração via YAML](#configuracao-via-yaml)
5. [Chaves suportadas](#chaves-suportadas)
6. [Interface de linha de comando](#interface-de-linha-de-comando)
7. [Dashboard interativo](#dashboard-interativo)
8. [Reprodutibilidade](#reprodutibilidade)
9. [Resultados gerados](#resultados-gerados)
10. [Métricas de calibração e validação](#metricas-de-calibracao-e-validacao)
11. [Pré-processadores e modelos suportados](#pre-processadores-e-modelos-suportados)
12. [Treinar modelo final e predições](#treinar-modelo-final-e-predicoes)
13. [Erros comuns e soluções](#erros-comuns-e-solucoes)
14. [Extensões e contribuições](#extensoes-e-contribuicoes)
15. [Rastreamento com MLflow](#rastreamento-com-mlflow)
16. [Testes](#testes)
17. [Docker](#docker)
18. [Empacotamento para distribuição](#empacotamento-para-distribuicao)
19. [Créditos/Autoria](#creditosautoria)

## Requisitos

Requer Python 3.9 ou superior. Principais dependências: `numpy`, `pandas`,
`scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `pyyaml`, `rich`, `joblib`,
`openpyxl` e `tqdm`. Pacotes opcionais incluem `xgboost`, `lightgbm`,
`catboost`, `torch`, `questionary`, `mlflow` e `dvc`.

Para exportar relatórios em PDF é necessário instalar dependências extras:

* `markdown` e `weasyprint` (recomendado), ou
* `markdown` e `pdfkit`, juntamente com o executável `wkhtmltopdf`.

## Instalação

Para utilizar o toolkit em modo de desenvolvimento instale no ambiente atual:

```bash
pip install -e .
```

Para habilitar os _hooks_ de formatação (`black` e `isort`) e lint (`flake8`) antes de cada commit,
instale o [pre-commit](https://pre-commit.com):

```bash
pre-commit install
```

Para uma instalação reprodutível, utilize:

```bash
pip install -r requirements-lock.txt
```

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
      method: [logreg, lda]

# Para evitar sobreajuste, mantenha `n_components` bem abaixo de
# `min(n_amostras - 1, n_variáveis)`; valores acima de 80% desse limite
# geram um aviso.

cv:
  random_state: 0
  strategies:
    - type: kfold
      n_splits: [5]
```

Para carregar múltiplos arquivos de uma pasta, utilize um padrão
glob. Por exemplo, para ler todos os arquivos Excel em
`data/RAW/`, configure:

```yaml
data:
  input_path: data/RAW/*.xlsx
  target_col: especie
```

* `preprocess.steps` lista os transformadores aplicados antes do classificador.
  Alguns pré-processos como `SG` aceitam parâmetros (janela, ordem do polinômio,
  derivada) que podem ser fixos ou listas para busca em grade. Nos arquivos de
  métricas, a coluna `preprocess_steps` registra apenas as etapas realmente
  empregadas pelo melhor modelo, incluindo os valores finais de seus
  hiperparâmetros.
* `models` define um ou mais estimadores; cada um pode receber um dicionário de
  hiperparâmetros (`grid`).
* `cv.strategies` especifica a estratégia de avaliação: `kfold`, `loo` ou
  `holdout`. É possível combinar múltiplas estratégias.

Para realizar validação cruzada respeitando grupos (por exemplo, diferentes
lotes ou indivíduos), informe a coluna de grupos em `data.group_col` e defina
`cv.cv_method: group`:

```yaml
data:
  input_path: dados/espectros.csv
  target_col: especie
  group_col: lote

cv:
  cv_method: group
  strategies:
    - type: kfold
      n_splits: [5]
```

### Chaves suportadas

#### data
- **input_path** (`str`): caminho para arquivo ou padrão glob de arquivos de
  entrada. Exemplos:
  `data/dados.xlsx` ou `data/RAW/*.xlsx` para múltiplos arquivos. Diretórios
  simples não são aceitos; se um diretório for informado, o `nircls` emitirá
  `DataError: Caminho(s) inválido(s)`.
- **x_sheet** (`str`, padrão `"X"`): nome da planilha com os espectros.
- **y_sheet** (`str`, padrão `"Y"`): nome da planilha com as classes.
- **target_cols** (`list[str]` | `str`): colunas alvo ou `"all"`. A chave
  **target_col** é um atalho para uma única coluna.
- **group_col** (`str`): nome da coluna que define grupos para validação
  cruzada agrupada.
- **spectral_range** (`[float, float]`): limites inferior e superior de
  comprimento de onda.
- **drop_nans** (`bool`, padrão `true`): remove linhas com valores ausentes.
- **out_dir** (`str`, padrão valor de `NIRCLS_RESULTS` ou `"results"`): diretório base de saída.

Os diretórios padrão para dados brutos e resultados podem ser ajustados
pelas variáveis de ambiente `NIRCLS_RAW` e `NIRCLS_RESULTS`.

#### preprocess
- **steps** (`list[str]`): lista de pré-processadores, por exemplo
  `SNV`, `MSC`, `DETREND`, `SG`, `EMSC`, `EPO`, `SCALE`.
- **sg.window**, **sg.polyorder**, **sg.deriv** (`int` ou `list[int]`): parâmetros
  do filtro Savitzky‑Golay. Listas ativam busca em grade.

#### cv
- **random_state** (`int`, padrão `42`): semente aleatória global.
- **inner_folds** (`int`): número de folds internos para validação aninhada.
- **cv_method** (`str`): `stratified`, `spxy` ou `group`.
- **strategies** (`list`): uma ou mais estratégias de avaliação:
  - `type: kfold` com chaves **n_splits** e opcional **n_repeats**;
  - `type: loo` (sem chaves adicionais);
  - `type: holdout` com **test_size**.

#### models
Cada item da lista define um estimador:
- **name** (`str`): identificador do modelo.
- **params** (`dict`): parâmetros fixos passados ao modelo.
- **grid** (`dict`): hiperparâmetros para busca em grade.
  Parâmetros que esperam tuplas (por exemplo, `hidden_layer_sizes` do `MLP`)
  devem ser representados como listas de listas no YAML para preservar a
  semântica de tupla, e.g.:

  ```yaml
  grid:
    hidden_layer_sizes:
      - [100]
      - [200]
  ```
- **calibration.method** (`"sigmoid"` ou `"isotonic"`) e
  **calibration.cv** (`int`): opções para calibração de probabilidades.

## Interface de linha de comando

O comando principal é `nircls` e suporta várias sub‑rotinas. As mensagens da CLI podem ser exibidas em português (`--lang pt`) ou inglês (`--lang en`, padrão). Para usuários leigos, basta executar `nircls run-all` para processar dados no diretório definido por `NIRCLS_RAW` (padrão `data/RAW`), treinar e avaliar o modelo padrão PLSDA. Para tentar todos os modelos disponíveis (requer dependências opcionais como `xgboost`, `lightgbm`, `catboost` e `torch`), utilize a flag `--all-models`. Essa opção ignora qualquer lista `models:` definida no arquivo de configuração e treina todos os modelos suportados. Para aplicar todos os tratamentos de pré-processamento (`SNV`, `MSC`, `DETREND`, `SG`, `EMSC`, `EPO`, `SCALE`), utilize `--all-preprocess`:

Antes de rodar qualquer comando `nircls`, crie e ative um ambiente virtual. Em sistemas Unix/macOS, utilize `python -m venv .venv` seguido de `source .venv/bin/activate`; no Windows, use `.venv\Scripts\activate`. Todos os comandos a seguir devem ser executados com o ambiente virtual ativo.

```bash
# Execução completa com parâmetros padrão (PLSDA)
nircls run-all

# Todos os modelos disponíveis (requer dependências opcionais)
nircls run-all --all-models

# Todos os modelos e pré-processamentos
nircls run-all --all-models --all-preprocess

# Mensagens em português
nircls run-all --lang pt

# Verificar a versão instalada
nircls --version

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

A verificação de conformidade com a norma ASTM para classificação pode ser
ativada com a flag `--astm` nos subcomandos `train` e `crossval`. O relatório
`astm_classification.json` informa se o conjunto possui ao menos 30 amostras,
se cada classe conta com 10 ou mais exemplos e se foi realizada validação
externa (holdout ou LOOCV). O resumo dos critérios é exibido no console com
marcas de verificação.

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

O dashboard permite executar todas as etapas do fluxo sem recorrer à CLI. A aba
**Preparar Dados** fornece análise exploratória básica (prévia das entradas,
distribuição de classes e curvas espectrais) antes do treinamento.

1. Inicie a aplicação:

   ```bash
   nircls dashboard --results results
   ```

   A pasta indicada armazenará os artefatos gerados.

2. Na aba **Preparar Dados**, envie o arquivo CSV/XLSX, escolha a coluna alvo e
   opcionalmente informe o intervalo espectral. Ao enviar o formulário a função
   `prepare_inputs` processa os dados e apresenta uma prévia, gráfico de barras
   com a contagem por classe e curvas espectrais (média ± desvio padrão).

3. Em **Treinar**, selecione o modelo desejado e forneça um grid de parâmetros
   em JSON. O botão *Treinar* chama `train_and_save` e grava o modelo treinado na
   pasta de resultados.

4. Em **Avaliar**, configure o número de folds, modelo e parâmetros. O formulário
   executa `run_multiple_eval` gerando métricas e relatórios. A seção também
   permite enviar um modelo salvo e um novo arquivo de dados para que
   `predict_with_model` produza o arquivo `predictions.csv`.

5. A aba **Resultados** lista todas as execuções presentes no diretório
   informado, apresenta uma tabela agregada de métricas e permite selecionar
   duas ou mais execuções para comparar métricas e figuras lado a lado.

Com essas etapas é possível conduzir um experimento completo apenas pela
interface web.

## Reprodutibilidade

Para permitir que outros reproduzam seus resultados:

* Defina a semente aleatória com o argumento global `--seed` ou pelo campo
  `cv.random_state` no arquivo de configuração.
* Registre a versão do software executando `nircls --version` ou consultando
  `nircls.__version__` em Python.
* Instale dependências a partir de `requirements-lock.txt` para garantir o
  mesmo ambiente:

  ```bash
  pip install -r requirements-lock.txt
  ```

* Para versionar dados utilize o [DVC](https://dvc.org). Após instalar o
  extra opcional com `pip install -e .[dvc]`, execute:

  ```bash
  dvc pull    # baixa os dados versionados
  dvc repro   # roda a pipeline definida em dvc.yaml
  dvc push    # envia novas versões para o remoto configurado
  ```

  Dessa forma, colaboradores podem obter exatamente a mesma versão dos dados e
  reproduzir o experimento localmente.

* Cite o projeto em publicações, por exemplo:

  > Dias, T. L. (2024). *nircls: NIR Species Classification Toolkit* (versão
  > 0.1.0) [software de computador]. Disponível no repositório GitHub do
  > projeto.

## Resultados gerados

Cada execução cria uma pasta em `results/<run_id>/<estratégia>/` onde
`<run_id>` identifica a data/hora do experimento. Dentro de cada estratégia são
produzidos:

* `metrics.json` – métricas detalhadas por partição;
* `metrics_summary.csv` – resumo das métricas agregadas;
* `report.md` – relatório científico em Markdown com tabelas, figuras e
  metadados de citação;
* `report.pdf` – versão em PDF quando `pandoc` ou `weasyprint` estiver
  disponível;
* `figures/` – gráficos de curva ROC, matriz de confusão e confiabilidade,
  além de `*.confusion.csv` com a matriz em formato tabular para análises
  externas.
* arquivos `shap_*.html` – figuras interativas de explicabilidade geradas com
  [SHAP](https://shap.readthedocs.io/) (quando a dependência opcional estiver
  instalada).
* `metadata.json` – informações de rastreabilidade como hashes dos arquivos de
  entrada, versões das dependências e semente aleatória.

O relatório é gerado inteiramente de forma local, sem qualquer integração com
serviços externos ou APIs. O `report.md` contém um cabeçalho YAML com título e
data do experimento, permitindo compartilhamento e citação direta em artigos ou
repositórios. Para distribuir os resultados, basta enviar o `report.md` junto com
a pasta `figures/` (que contém as imagens referenciadas no relatório) ou a
versão PDF (`report.pdf`). O cabeçalho YAML traz título, data e versão do
projeto, facilitando citações padronizadas.

## Métricas de calibração e validação

Os arquivos de métricas contêm resultados para calibração (dados de treino) e
validação/teste. As métricas de treino recebem o prefixo `train_` e podem incluir
`accuracy`, `f1_macro`, `balanced_accuracy`, `brier_score` e o erro de calibração
esperado (`ece`). A comparação entre `train_` e métricas sem prefixo ajuda a
avaliar overfitting e o desempenho em dados não vistos.
Sempre que a diferença entre `train_BalancedAcc_mean` e `BalancedAcc_mean`
ultrapassa 0.05, a ferramenta marca `overfit_warning = True` em `metrics.json`
e no relatório científico, sinalizando possível sobreajuste.

## Teste Y-scrambling

O *Y-scrambling* agora é executado automaticamente para todos os modelos. Os
rótulos são embaralhados diversas vezes e o modelo é re‑treinado em cada
permuta, gerando uma distribuição de desempenhos ao acaso. As colunas
`Yscramble_mean` e `Yscramble_p95` em `metrics_summary.xlsx` e `metrics.json`
trazem respectivamente a média e o percentil 95% dessa distribuição.

Se o desempenho obtido no conjunto original ficar próximo ou abaixo desses
valores, é um forte indicativo de que o modelo está ajustando apenas ruído.

Por padrão são executadas 20 permutações. Para reduzir o tempo de execução,
ajuste o número de iterações com `y_scramble_n_iter` (por exemplo, 5 para um
teste mais rápido). Defina `y_scramble_n_iter: 0` ou `overfit_check: false`
para desativar totalmente o Y‑scrambling:

```yaml
models:
  - name: PLSDA
    grid: {n_components: [2]}
    y_scramble_n_iter: 5  # menos iterações (0 desativa)
    # overfit_check: false
```

## Pré-processadores e modelos suportados

Pré-processadores disponíveis:

* `SNV` – Standard Normal Variate
* `SG` – Filtro Savitzky‑Golay
* `MSC` – Multiplicative Scatter Correction
* `Detrend` – remoção de tendências polinomiais
* `EMSC` – Extended MSC
* `EPO` – External Parameter Orthogonalization
* `SCALE` – padronização com `StandardScaler`

Modelos implementados (modelos que dependem de pacotes extras são ignorados se as bibliotecas não estiverem instaladas):

* `PLSDA` – usa regressão de mínimos quadrados parciais para discriminação,
  adequada a dados de alta dimensionalidade com colinearidade. Após a redução
  dimensional, pode empregar diferentes classificadores (`logreg`, `lda`,
  `qda`, `svm`, `knn`). Hiperparâmetros desses métodos podem ser ajustados via
  `grid`. Valores de `n_components` acima de 80% de
  `min(n_amostras - 1, n_variáveis)` podem levar a sobreajuste e geram um
  aviso. Exemplos de configuração:

  ```yaml
  models:
    - name: PLSDA
      params: {method: qda}
      grid: {n_components: [2, 4], reg_param: [0.0, 0.1]}
    - name: PLSDA
      params: {method: knn}
      grid: {n_neighbors: [3, 5], n_components: [2]}
  ```

  adequada a dados de alta dimensionalidade com colinearidade. Valores de
  `n_components` acima de 80% de `min(n_amostras - 1, n_variáveis)` podem
  levar a sobreajuste e geram um aviso. O parâmetro `method` escolhe o
  classificador final (`"logreg"` ou `"lda"`) e pode ser avaliado via grade,
  por exemplo: `grid: {n_components: [2], method: [logreg, lda]}`.

* `SVM_RBF` e `SVM_LINEAR` – máquinas de vetores de suporte; kernel RBF modela
  fronteiras não lineares enquanto o kernel linear serve para separações simples
* `RF` (Random Forest) – conjunto de árvores de decisão construídas por bootstrap
  que reduz o overfitting
* `GB` (Gradient Boosting) – adiciona modelos fracos sequencialmente para
  corrigir erros residuais e capturar padrões complexos
* `LOGREG` (Regressão Logística) – modelo linear probabilístico para
  classificação binária ou multiclasse
* `MLP` (rede neural Multi‑Layer Perceptron) – camadas densas capazes de
  aprender relações não lineares. A grade padrão avalia `max_iter` em
  `[500, 1000]` e ativa `early_stopping` para mitigar avisos de não
  convergência sem aumentar excessivamente o tempo de treinamento.
* `DT` (Árvore de decisão) – estrutura hierárquica que particiona os dados por
  atributos, fácil de interpretar mas sensível a ruído
* `KNN` – atribui o rótulo mais comum entre os vizinhos mais próximos,
  dispensando fase de treinamento explícita
* `ExtraTrees` (Extremely Randomized Trees) – florestas com divisões totalmente aleatórias,
  indicadas para reduzir variância em dados ruidosos
* `AdaBoost`, `XGBoost` (`xgboost`), `LightGBM` (`lightgbm`), `CatBoost` (`catboost`) – métodos de boosting robusto que capturam
  relações complexas e lidam bem com grandes conjuntos de dados
* `QDA` (Quadratic Discriminant Analysis) – adequado quando cada classe tem matrizes de
  covariância distintas. Utilize o parâmetro de regularização `reg_param` (>0) para
  atenuar problemas de posto da matriz de covariância em dados de alta
  dimensionalidade.
* `Naive Bayes` – assume independência condicional entre atributos, servindo como baseline
  rápido em alta dimensionalidade
* `CNN 1D` e `LSTM 1D` – redes neurais para séries 1D; convoluções extraem padrões locais
  enquanto LSTMs modelam dependências de longo prazo (dependem de `torch`)

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
* **Diretório informado em `data.input_path`** – a chave deve apontar para
  arquivos ou padrões glob (ex.: `data/RAW/*.xlsx`). Passar um diretório simples
  gera `DataError: Caminho(s) inválido(s)`.
* **Modelo incompatível** – utilize um nome de modelo suportado e revise os parâmetros.

## Extensões e contribuições

O toolkit foi projetado para ser extensível. Novos pré‑processadores ou modelos
podem ser adicionados implementando classes compatíveis com `scikit-learn` e
registrando‑as nos módulos apropriados. Contribuições são bem‑vindas via pull
requests.

### Plugins de modelos

Desde a versão atual, modelos podem ser fornecidos por pacotes externos usando
``entry_points`` do Python. Basta criar uma classe que implemente
``BaseModelBuilder`` e registrá‑la em ``pyproject.toml``:

```python
# my_plugin/builder.py
from nircls.models.base import BaseModelBuilder
from sklearn.dummy import DummyClassifier

class DummyBuilder(BaseModelBuilder):
    name = "DUMMY"
    description = "Dummy classifier"

    def build(self, grid):
        return DummyClassifier()

    def default_grid(self):
        return {}
```

No arquivo ``pyproject.toml`` do plugin, exponha a classe no grupo
``nircls.model_builders``:

```toml
[project.entry-points."nircls.model_builders"]
dummy = "my_plugin.builder:DummyBuilder"
```

Após instalar o pacote, o comando ``nircls list-models`` mostrará o novo modelo
e ele poderá ser referenciado nas configurações YAML ou no assistente
``nircls wizard``.

Os resultados serão salvos em `results/<run_id>/<estratégia>/`. Cada pasta de
estratégia contém as métricas (`metrics.json`, `metrics_summary.xlsx`), as figuras
geradas e um relatório final em Markdown (`report.md`).

## Rastreamento com MLflow

Para registrar experimentos com [MLflow](https://mlflow.org) instale a
dependência opcional:

```bash
pip install -e .[tracking]
```

Em seguida, inicie um servidor local:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Execute os comandos do `nircls` apontando para o URI do servidor:

```bash
nircls train --config experimento.yml --mlflow http://localhost:5000
# ou rode validação cruzada
nircls crossval --config experimento.yml --mlflow http://localhost:5000
```

Abra `http://localhost:5000` no navegador para consultar o histórico de
execuções e artefatos registrados.

Quando o extra `dvc` está instalado, o comando `train` registra no MLflow o
hash dos dados versionados pelo DVC, permitindo correlacionar código, dados e
resultados em um único lugar.

## Testes

Para rodar a suíte de testes utilize o [pytest](https://pytest.org):

```bash
pytest
```

Para validar as anotações de tipo estático, execute o
[mypy](https://mypy-lang.org/):

```bash
mypy .
```

Se nenhum erro for exibido, todas as verificações passaram. Mensagens
apontam os arquivos e linhas que precisam de correção.



## Docker

Para construir a imagem:

```bash
docker build -t nircls .
```

Para iniciar o dashboard:

```bash
docker run -p 8501:8501 nircls
```

Opcionalmente, publique a imagem em um registro interno para facilitar a distribuição no laboratório.

## Empacotamento para distribuição

Para gerar um pacote independente com o código e as configurações do `nircls`, execute:

```bash
python make_nircls_zip.py
```

Será criado o arquivo `nircls_bundle.zip` contendo `src/nircls`, exemplos de configuração em `configs/` e um `requirements.txt` gerado a partir do ambiente atual.

Em outra máquina:

1. Extraia o `nircls_bundle.zip`.
2. Crie um ambiente virtual e instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```
3. Utilize os utilitários do `nircls` a partir do diretório extraído.

## Créditos/Autoria

Desenvolvido por Thalles Loiola Dias, Discente da UFLA.

## Como citar

Se utilizar este toolkit em trabalhos acadêmicos ou publicações, cite conforme o arquivo [CITATION.cff](CITATION.cff).

## Licença

Distribuído sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais informações.

