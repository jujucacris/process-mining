#Codigo baseado em : https://github.com/tnolle/binet
class DAE:

    def __init__(self,params):
        """
        self.parametros deve ser um diccionario contendo:
            fator,no,nitmax
        """
          
        #self.fator = params.pop('fator') #vai ser usado no calculo dos neuronios da cadama oculta(no=ne * fator)
        self.no = params.pop('no') #numero de neuronios na camada oculta
        self.nitmax= params.pop('nitmax') #numero de iterações maximo (epocas)
        self.modelo = []       
        self.nro_camadas_ocultas = params.pop('nro_camadas_ocultas') #FIX parameter not used in model_fn yet
        self.ruido_desv_padrao = params.pop('ruido_desv_padrao')    
    
    @staticmethod
    def model_fn(self,features):
        
        # Importar Keras
        from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise #FIX
        from tensorflow.keras.models import Model #FIX
        from tensorflow.keras.optimizers import Adam #FIX
    
        # Numero de entradas ou numero de neuronios na camada de entrada(ne)
        ne = features.shape[1] 
    
        # Camada de entrada
        input = Input(shape=(ne,), name='input')
        x = input
    
       # Noise layer
        #if noise is not None:
        x = GaussianNoise(self.ruido_desv_padrao)(x)
            
        # Camada Oculta
        for i in range(self.nro_camadas_ocultas):
#            if(self.fator is not None):
#                self.no=ne * self.fator #Numero de neuronios usados na camada oculta    
            x = Dense(int(self.no[i]), activation='relu', name=f'hid{i + 1}')(x)
            x = Dropout(0.5)(x)
    
        # Camada de saida
        output = Dense(ne, activation='sigmoid', name='output')(x)
    
        # Configurar modelo
        modelo = Model(inputs=input, outputs=output)
    
        # Compilar Modelo
        modelo.compile(
            optimizer=Adam(lr=0.0001, beta_2=0.99),
            loss='mean_squared_error',
        )
    
        return modelo
    
    def treinar(self, Xtr, Ytr, Xval, Yval):
        """
        # Xtr: conjunto de treinamento
        # Ytr: labels do conjunto de treinamento
        # XVal: conjunto de validacao
        # YVal: labels do conjunto de validacao
        """                                    
        # Criar modelo  
        self.modelo = self.model_fn(self,Xtr)
        
        #Treinar modelo
        self.modelo.fit(Xtr, Xtr, epochs=self.nitmax, validation_data=(Xval,Yval), batch_size=500) 
            
            
    def test(self,Xtest, Ytest):
                
        # Gerar predicoes
        predictions = self.modelo.predict(Xtest)            
        return predictions

  