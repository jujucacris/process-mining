# process-mining

Projeto para a disciplina de Mineração de Dados do 2o semestre de 2019 na EACH - USP.

##  CONSIDERAÇÕES

Dentro da pasta MLP_Python encontram-se todos os códigos criados para os experimentos descritos no relatório.
O arquivo experimentos.py é o principal arquivo utilizado para a execução dos experimentos, e é onde estão contidas
as parametrizações a chamadas às classes do autoencoder. Este arquivo já está configurado para a execução de experimentos
simples de testes.
O arquivo clase_MLP.py possui a lógica núcleo do autoencoder criada pelo grupo, e é chamado a partir do arquivo
main_autoencoder.py.
O arquivo main_autoencoder.py possui as estratégias de parametrização e cálculo dos erros de reprodução para
a classificação final do algoritmo e também as chamadas aos scripts de processamentos gráficos e cálculos de estatísticas
relacionados ao pós processamento.
Os arquivos autoencoder_keras.py e autoencoder_nolle.py são os utilizados para comparação com os nossos experimentos.


## EXPERIMENTOS

### Bibliotecas

Para a execução dos experimentos, é necessário ter instaladas no ambiente as seguintes bibliotecas:
   - pandas
   - numpy
   - sklearn
   - keras
   - matplotlib
   - tensorflow

### Execução

Pode ser utilizado tanto um IDE quanto um terminal para, de dentro da pasta MLP_Python, simplesmente executar o arquivo
'experimentos.py', ou na linha de comando como segue:
``python3 experimentos.py`
Os resultados resumidos da execução do experimento s`erão impressos na console, e os arquivos com os resultados extensivos
da execução podem ser encontrados na pasta /MLP_python/resultados, como segue:
   - exp(nro_experimento)_curva_precision_recall_(nome_dataset).png -> curva precision-recall
   - exp(nro_experimento)_curva_roc_(nome_dataset).png -> curva ROC
   - Exp(nro_experimento)_EQMs_nit.csv -> erro quadrático médio a cada iteração do cross validation
   - exp(nro_experimento)_evolucao_EQM_(nome_dataset).png -> gráfico da evolução do EQM
   - exp(nro_experimento)_matriz_confusao_nao_normalizada_0.png -> matriz de confusão não normalizada para cada iteração do cross validation
   - exp(nro_experimento)_matriz_confusao_normalizada_0.png -> matriz de confusão normalizada para cada iteração do cross validation
   - exp(nro_experimento)_matrizes_saida_(nome_dataset).csv -> csv com as matrizes de confusão de cada iteração e resumo com totais e médias na última linha