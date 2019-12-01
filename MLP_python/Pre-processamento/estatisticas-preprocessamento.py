#!/usr/bin/python
# -*- coding: utf-8 -*-

#*******************************************
# ANALISE DESCRITIVO DOS DADOS E
# PRE-PROCESSAMENTO
# v1.0 - main
#*******************************************
from cPreprocessor import cPreprocessor as cPreprocessor
import os

# Ler arquivo do log
oPreprocessor = cPreprocessor(os.path.join(Conversor de JSON","p2p-0.3-1-usuarios.csv"))
oPreprocessor.LerDataset()

# Visualizar os primeiros registros do log
oPreprocessor.dataset_log.head()
oPreprocessor.dataset_log.describe()

# Analise descritivo do log sem preprocesamento
descricao_dataset_log = oPreprocessor.AnaliseDescritivo()

# Conversao o log em dataset de traces
data_traces = oPreprocessor.Converte_log_em_traces()

# Gera��o de estat�sticas do log
descricao_dataset_trace = oPreprocessor.Descricao_dataset_traces()
print(descricao_dataset_trace)

# Gera��o de graficas descritivas
oPreprocessor.Gerar_graficos()
