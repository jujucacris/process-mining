import numpy as np
import pandas as pd

class DAE:

    def __init__(self,params):
        self.config = params # no: Numero de neuronios na camada oculta
    
    def model_fn(dataset):
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise
        from keras.models import Model
        from keras.optimizers import Adam
    
        hidden_layers = 5
    
        features = dataset
    
        # Parameters
        input_size = features.shape[1]
    
        # Input layer
        input = Input(shape=(input_size,), name='input')
        x = input
    
        # Hidden layers
        x = Dense(hidden_layers, activation='relu', name=f'hid{i + 1}')(x)
    
        # Output layer
        output = Dense(input_size, activation='sigmoid', name='output')(x)
    
        # Build model
        model = Model(inputs=input, outputs=output)
    
        # Compile model
        model.compile(
            optimizer=Adam(lr=0.0001, beta_2=0.99),
            loss='mean_squared_error',
        )
    
    #    return model, features, features
        return model, features
    
    def treinar(self, Xtr, Ytr, Xval, Yval):
            """
            Calculate the anomaly score for each event attribute in each trace.
            Anomaly score here is the mean squared error.
    
            :param traces: traces to predict
            :return:
                scores: anomaly scores for each attribute;
                                shape is (#traces, max_trace_length - 1, #attributes)
    
            """                                    
            # Criar modelo
            _, features, _ = self.model_fn(Xtr, **self.config)
            
            #Treinar modelo
            self.model.fit(Xtr, Xtr, epochs=50, batch_size=500) #FIX fazer parametros de entrada da classe
            
            
    def test(self,Xtest, Ytest):
                
        # Gerar predicoes
        predictions = self.model.predict(Xtest)            
        return predictions

  