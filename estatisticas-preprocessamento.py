# importar librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# leer arquivo do log
data_act = pd.read_csv("p2p-0.3-4.csv")

# leer arquivo do log com representacao
#data_nolle = pd.read_csv("p2p-0.3-4-nolle.csv")

# ---------------------------------------
# Estatisticas de contagem geral no log
# ---------------------------------------

# Visualizar os primeiros registros do log
data_act.head()

# Total de eventos no log
data_act.count()
len(data_act)

# Total de cases no log
data_act['traceid'].max()

# Quantidade de Null no log
data_act.isnull().any()

# Total ocorrencias por atividade
data_act['activity'].value_counts()
data_act['activity'].value_counts(normalize=True) # frequencia relativa (porcentagem)

# Total ocorrencias por tipo de case
data_act['label'].value_counts()
data_act['label'].value_counts(normalize=True) # frequencia relativa (porcentagem)

# Total ocorrencias por usuario
data_act['user'].value_counts()
data_act['user'].value_counts(normalize=True) # frequencia relativa (porcentagem)

# Visualizar os eventos no case
data_trace = log_to_traces(data_act)
data_trace.head()

# Numero maximo de eventos por case
data_trace['nr_events'].max()

# Numero mi­nimo de eventos por case
data_trace['nr_events'].min()

# Variacoes no log
data_trace_variant = data_trace.groupby(['trace'], sort = True)
total_variations = data_trace_variant['trace'].agg([np.count_nonzero])
total_variations.describe()

# ---------------------------------------
# Graficos
# ---------------------------------------

# Frequencia de traces anomalos e normal
label = data_trace['label'].value_counts().index
frequencia = data_trace['label'].value_counts().values

plt.bar(label,frequencia, color='y')
plt.title('Grafico barras')
plt.xlabel('label', color='y')
plt.ylabel('frequencia', color='r')
plt.show()
plt.savefig('barras.png', bbox_inches='tight')

# Frequencia de eventos por trace
data_trace['nr_events'].plot.hist()
plt.savefig('histograma.png', bbox_inches='tight')

# quantidade de eventos por traces
plt.scatter(data_trace['traceid'], data_trace['nr_events'], label = 'Pontos', color = 'r', marker = '*', s = 6)
plt.legend()
plt.show()
plt.savefig('scatter.png', bbox_inches='tight')

#  Gráficos boxplot - O retângulo é formado por três Quartis que dividem o dados em quatro rols com 25% dos dados cada.
sns.set(style="whitegrid", color_codes=True)
sns.boxplot(data=total_variations)
plt.savefig('boxplot.png', bbox_inches='tight')

# ---------------------------------------
# Funcoes
# ---------------------------------------

def log_to_traces(data_log):
    data_traces = pd.DataFrame(columns = ['traceid','trace','nr_events','label'])
    trace = ''
    traceid = 1
    nr_events = 0

    for index, row in data_log.iterrows():
        if (row['traceid'] ==  traceid):
            trace = trace + row['activity'] + ','
            nr_events += 1
        else:
            data_traces.loc[traceid-1] = [traceid, trace[:-1], nr_events, row['label']]
            traceid += 1
            trace=''
            nr_events=0
    data_traces.loc[traceid-1] = [traceid, trace[:-1], nr_events, row['label']]
    return data_traces


