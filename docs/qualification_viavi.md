# Dados VIAVI para qualificação — integração reproduzível

O adaptador `scripts/prepare_viavi_qualification.py` lê um CSV previamente padronizado com as colunas:
`status_espectral`, `grupo_provisorio_A`, `nome_popular_padronizado` e
125 colunas espectrais com prefixo `nm_`. O arquivo XLSX bruto **não é alterado**.

Exemplo:
```bash
python scripts/prepare_viavi_qualification.py \
  --input "/caminho/privado/NIR_VIAVI_metadados_padronizados.csv" \
  --output-dir "./data/RAW"
nircls crossval --config configs/qualification_viavi.yaml
```

O comando gera quatro CSVs, incluindo dados espectrais por leitura e médias por identificador.
A configuração de exemplo utiliza `data.group_col: grupo_codigo` para impedir a
distribuição de leituras do mesmo código entre partições. Os resultados do toolkit
devem ser reexecutados e auditados; os códigos da coluna A ainda são **grupos
provisórios**, não necessariamente peças físicas independentes.

### Limitações
- `nome_popular` é rótulo comercial **provisório** e não substitui a
  identificação anatômica. Não divulgar resultados como acurácia botânica.
- A regra de inclusão de `>=8` códigos por classe é uma configuração
  exploratória, sujeita a revisão antes da análise confirmatória.
- Não misturar leituras brutas de vários equipamentos em um único modelo.
- Todos os arquivos espectrais e identificações continuam privados e locais:
  **não adicionar arquivos de dados ao repositório**.
- O exemplo de YAML ainda não passou por teste de integração completo;
  conferir compatibilidade com a versão instalada de `nircls`.
