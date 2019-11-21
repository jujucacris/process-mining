def grafica_evolucao_EQM(vet_erro_tr,vet_erro_val):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(1)
    fig.set_size_inches(10,10)
    plt.xlabel('Numero de epocas')
    plt.ylabel('EQM(Função de perda)')
    #plt.grid(True)
    plt.plot(range(0,len(vet_erro_tr)), vet_erro_tr, label='Treinamento')
    plt.plot(range(0,len(vet_erro_val)), vet_erro_val, label='Validação')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('grafica_evolucao_EQM.png', bbox_inches='tight')