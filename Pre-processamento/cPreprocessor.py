#!/usr/bin/python
# -*- coding: utf-8 -*-

#*******************************************
# ANALISE DESCRITIVO DOS DADOS E
# PRE-PROCESSAMENTO
# v1.0 - classe
#*******************************************

# importar librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from sklearn.preprocessing import LabelEncoder

#borrar
#dataset_log = pd.read_csv("~/GIT/PM2/process-mining/Conversor de JSON/p2p-0.3-1.csv")
#dataset_log['label'] = pd.get_dummies(dataset_log['label']).loc[:,'normal']
#dataset_trace = dataset_log.groupby('traceid').apply(f)

#--
# Funcao para determinar criterios de grupos dos traces
def f(x):
    return pd.Series(dict(trace = "{%s}" % ','.join(x['activity']),
                          users = "{%s}" % ','.join(x['user']),
                          nr_events = x['activity'].count(),
                          nr_users = x['user'].nunique(),
                          label = x['label'].mean()))

class cPreprocessor(object):
    def __init__(self,origem):
        self.origem = origem
        self.dataset_trace = []
        self.dataset_log = []

    def LerDataset(self):
        self.dataset_log = pd.read_csv(self.origem)
        self.dataset_log['label'] = pd.get_dummies(self.dataset_log['label']).loc[:,'normal']# convertir a binaria, classe possitiva = "anomaly"


    def AnaliseDescritivo(self):
        # Nro de eventos no log
        nro_eventos = self.dataset_log.shape[0]

        # Total de traces no log
        nro_traces = self.dataset_log['traceid'].max()

        # Numero de atividades diferentes
        #nro_act_diferentes = pd.DataFrame([data_act['activity'].value_counts(), data_act['activity'].value_counts(normalize=True)*100])
        nro_act_diferentes = pd.DataFrame([self.dataset_log['activity'].value_counts(), self.dataset_log['activity'].value_counts(normalize=True)*100])


        # Distribuicao das classes
        classes = self.dataset_log['label'].value_counts(normalize=True)*100 # frequencia relativa em porcentagem
        classes = np.array(classes)
        classes = 'Classe Normal: {0}, Classe Anomalo: {1}'.format(str(round(classes[0],2)),str(round(classes[1],2)))

        # Numero de usuarios envolvidos nu trace
        nro_user_difetentes = pd.DataFrame([self.dataset_log['user'].value_counts(), self.dataset_log['user'].value_counts(normalize=True)*100])

        # Quantidade de Null no log
        tem_null = self.dataset_log.isnull().any().any()

        return np.array([["Nro eventos", nro_eventos],
                        ["Nro traces", nro_traces],
                        ["Nro act diferentes", nro_act_diferentes.shape[0]],
                        ["Classes", classes],
                        ["Usuarios", nro_user_difetentes.shape[0]],
                        ["Tem null", tem_null]])


    # Funcao para convertir o log em traces
    def Converte_log_em_traces(self):
        self.dataset_trace = self.dataset_log.groupby('traceid').apply(f)
        return self.dataset_trace

    def Descricao_dataset_traces(self):
        # Numero maximo de eventos por trace
        nro_max_eventos = self.dataset_trace ['nr_events'].max()

        # Numero mi­nimo de eventos por trace
        nro_min_eventos = self.dataset_trace ['nr_events'].min()

        # Numero de traces longos
        traces_longos = self.dataset_trace[self.dataset_trace ['nr_events'].isin(range(10,15))]
        nro_traces_longos = traces_longos['trace'].count()

        # Numero de traces longos anomalos
        traces_longos_anomalos = traces_longos.groupby('label')
        traces_longos_anomalos = traces_longos_anomalos.get_group(0.0)['trace'].count()

        # Numero de traces anomalos
        traces_anomalos = self.dataset_trace.groupby('label')
        traces_anomalos = traces_anomalos.get_group(0.0)['trace'].count()

        # Numero maximo de usuarios por trace
        nro_max_usuarios = self.dataset_trace ['nr_users'].max()

        # Numero mi­nimo de usuarios por trace
        nro_min_usuarios = self.dataset_trace ['nr_users'].min()

        # Variacoes no log
        nro_variacoes_trace = self.dataset_trace['trace'].nunique()

        return np.array([["Nro max eventos", nro_max_eventos],
                        ["Nro min eventos", nro_min_eventos],
                        ["Nro max usuarios", nro_max_usuarios],
                        ["Nro min usuarios", nro_min_usuarios],
                        ["Nro variacoes trace", nro_variacoes_trace],
                        ["Nro traces 'longos'", nro_traces_longos],
                        ["Nro traces 'longos' anomalos", traces_longos_anomalos],
                        ["Nro traces anomalos", traces_anomalos]])

    # Gerar os gráficos descritivos
    def Gerar_graficos(self):
        # Grafico de barras da distribuição das classes
        plt.figure()
        plt.bar(['normal','anomalo'],self.dataset_trace['label'].value_counts().values, color='b')
        plt.title('Distribuição das classes no conjunto de dados')
        plt.xlabel('Label', color='b')
        plt.ylabel('Frequencia', color='b')
        #plt.show()
        plt.savefig('distribuicao_classes.png', bbox_inches='tight')

        # Distribuicao dos traces segundo seu tamanho
        plt.figure()
        plt.hist(self.dataset_trace['nr_events'], color='y')
        plt.title('Distribuição dos traces segundo seu tamanho')
        plt.xlabel('Tamanho do trace', color='y')
        plt.ylabel('Frequencia', color='y')
        #plt.show()
        plt.savefig('distribuicao_traces_tamanho.png', bbox_inches='tight')

        # Distribuicao dos traces segundo os usuarios envolvidos nele
        plt.figure()
        plt.hist(self.dataset_trace['nr_users'], color='y')
        plt.title('Distribuição dos traces segundo usuarios envolvidos')
        plt.xlabel('Quantidade de usuários envolvidos no trace', color='y')
        plt.ylabel('Frequencia', color='y')
        #plt.show()
        plt.savefig('distribuicao_traces_usuarios.png', bbox_inches='tight')

        #  Gráficos boxplot dos 3 atributos das classes
        plt.figure()
        variacoes_traces =  self.dataset_trace.drop_duplicates(['trace'], keep='last')
        sns.set(style="whitegrid", color_codes=True)
        sns.boxplot(data=variacoes_traces)
        plt.title('Distribuição dos tipos de traces')
        plt.savefig('boxplot_eventos_usuarios.png', bbox_inches='tight')

        # Distribução das variações de traces
        plt.figure()
        sns.set(style="whitegrid", color_codes=True)
        sns.boxplot(data=self.dataset_trace['trace'].value_counts())
        plt.title('Distribuição das variacoes do traces no log')
        plt.savefig('boxplot_variacoes_traces.png', bbox_inches='tight')
