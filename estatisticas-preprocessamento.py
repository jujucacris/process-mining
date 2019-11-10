# importar librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# leer arquivo do log
data_act = pd.read_csv("Conversor de JSON/p2p-0.3-1.csv")

# leer arquivo do log com representacao
#data_nolle = pd.read_csv("p2p-0.3-4-nolle.csv")

# ---------------------------------------
# Estatisticas de contagem geral no log
# ---------------------------------------

# Visualizar os primeiros registros do log
data_act.head()
data_act.describe()

# Total de eventos no log
data_act.count()
len(data_act)

# Total de cases no log
data_act['traceid'].max()

# Quantidade de Null no log
data_act.isnull().any()

# Total ocorrencias por atividade
data_describe_activities = pd.DataFrame(data_act['activity'].value_counts())
data_describe_activities['porcentagem'] = pd.DataFrame(data_act['activity'].value_counts(normalize=True))

# Total ocorrencias por tipo de case
data_act['label'].value_counts()
data_act['label'].value_counts(normalize=True) # frequencia relativa (porcentagem)

# Total ocorrencias por usuario
data_act['user'].value_counts()
data_describe_usuarios = pd.DataFrame(data_act['user'].value_counts())
data_describe_usuarios['porcentagem'] = pd.DataFrame(data_act['user'].value_counts(normalize=True))

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

# Grafico de barras empilhadas para ver quais atividades aparecem em traces anomalos ou normal
tabela = pd.pivot_table(data=data_act, values='traceid', index='activity', columns='label', aggfunc='count')
tabela[np.isnan(tabela)] = 0

##$$$$

# Ajusta o espaço entre os dois gráficos
plt.subplots_adjust(wspace=1)
plt.show()


# Frequencia de eventos por trace
data_trace['nr_events'].plot.hist()
plt.savefig('histograma.png', bbox_inches='tight')

# Scatter de traces, quantidade de eventos e categoría
tabela_trace = data_trace.drop_duplicates(['trace'], keep='last')
label_encoder = LabelEncoder()
tabela_trace['label'] = label_encoder.fit_transform(tabela_trace['label'])

scatter_plot = plt.scatter(tabela_trace['nr_events'], tabela_trace['nr_distinct_users'], alpha=0.5, c = tabela_trace['label'], s = 6)
plt.legend('eventos, usuarios')
plt.xlabel('Numero de eventos no trace')
plt.ylabel('Numero de usuarios distintos no trace')
plt.show()
plt.savefig('scatter.png', bbox_inches='tight')


y = tabela_trace['label']
X = tabela_trace.ix[:, 'nr_events':]
plt.scatter(X[y==0]['traceid'], X[y==0]['nr_distinct_users'], label='normal', s= np.pi, c='red',alpha=0.5)
plt.scatter(X[y==1]['traceid'], X[y==1]['nr_distinct_users'], label='anomaly', s= np.pi, c='blue',alpha=0.5)
plt.legend('eventos, usuarios')
plt.xlabel('Numero de eventos no trace')
plt.ylabel('Numero de usuarios distintos no trace')
plt.show()




#  Gráficos boxplot - O retângulo é formado por três Quartis que dividem o dados em quatro rols com 25% dos dados cada.
sns.set(style="whitegrid", color_codes=True)
sns.boxplot(data=total_variations)
plt.savefig('boxplot.png', bbox_inches='tight')

# ---------------------------------------
# Funcoes
# ---------------------------------------

def log_to_traces(data_log):
    data_traces = pd.DataFrame(columns = ['traceid','trace', 'users','nr_events','nr_distinct_users','label'])
    trace = []
    users = []
    traceid = 1
    nr_distinct_users = 0

    for index, row in data_log.iterrows():
        if row['traceid'] ==  traceid:
            trace.append(row['activity'])
            if row['user'] not in users:
                nr_distinct_users += 1
            users.append(row['user'])
        else:
            data_traces.loc[traceid-1] = [traceid, ",".join(trace), ",".join(users), len(trace), nr_distinct_users, row['label']]
            traceid += 1
            nr_distinct_users = 0
            trace = []
            users = []
    data_traces.loc[traceid-1] = [traceid, ",".join(trace), ",".join(users), len(trace), nr_distinct_users, row['label']]
    return data_traces


data_trace = log_to_traces(data_act)