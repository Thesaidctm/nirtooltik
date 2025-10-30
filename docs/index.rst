Welcome to nircls's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api


Preprocessamento e testes isolados
----------------------------------

O arquivo de exemplo :code:`configs/eucalipto_spxy.yaml` habilita o
campo :code:`preprocess.evaluate_individual`. Quando definido como
``true`` e a lista :code:`steps` contém múltiplos itens, o ``nircls``
gera um pipeline separado para cada transformador em vez de encadeá-los
de imediato. Essa estratégia facilita testar o impacto de cada etapa no
desempenho dos modelos.

.. code-block:: yaml

   preprocess:
     evaluate_individual: true
     steps: [SNV, MSC, AQS, SG, SCALE]

Execute o arquivo diretamente para conferir o comportamento:

.. code-block:: bash

   nircls run-all --config configs/eucalipto_spxy.yaml

O transformador ``AQS`` (Adaptive Quantile Segment Normalization) aplica uma
normalização patenteável baseada em segmentação adaptativa por derivadas e
ajuste de quantis, combinando com uma etapa de mistura ponderada pela
estabilidade espectral. Os parâmetros ``min_segment_size`` e
``blend_strength`` podem ser ajustados com as chaves ``aqs.min_segment_size`` e
``aqs.blend_strength`` dentro do arquivo YAML de configuração.

Os relatórios criados pelo comando listam modelos como
``PLSDA__snv`` e ``RF__msc``, sinalizando o pré-processamento avaliado
isoladamente. Para voltar ao encadeamento completo, defina o campo como
``false`` ou remova-o do YAML.
