#!/usr/bin/python
# -*- coding: utf-8 -*-

#*******************************************
# ANALISE DESCRITIVO DOS DADOS E
# PRE-PROCESSAMENTO
# v1.0 - main
#*******************************************
from cPreprocessor import cPreprocessor as cPreprocessor

# Ler arquivo do log
oPreprocessor = cPreprocessor("~/GIT/PM2/process-mining/Conversor de JSON/p2p-0.3-1.csv")
oPreprocessor.LerDataset()

# Visualizar os primeiros registros do log
oPreprocessor.dataset_log.head()
oPreprocessor.dataset_log.describe()

# Analise descritivo do log sem preprocesamento
descricao_dataset_log = oPreprocessor.AnaliseDescritivo()

# Conversao o log em dataset de traces
data_traces = oPreprocessor.Converte_log_em_traces()

# Geração de estatísticas do log
descricao_dataset_trace = oPreprocessor.Descricao_dataset_traces()

# Geração de graficas descritivas
oPreprocessor.Gerar_graficos()


