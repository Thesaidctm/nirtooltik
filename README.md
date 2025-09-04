# nircls — NIR Species Classification Toolkit

Instalação: `pip install -e .`

## Executando avaliações

Após preparar um arquivo de configuração, execute a validação cruzada pelo CLI:

```bash
nircls crossval --config caminho/para/config.yml
```

Os resultados serão salvos em `results/<run_id>/<estratégia>/`. Cada pasta de
estratégia contém as métricas (`metrics.json`, `metrics_summary.csv`), as figuras
geradas e um relatório final em Markdown (`report.md`).
