def grafica_evolucao_EQM(vet_erro_tr,vet_erro_val,nome,j):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    fig = plt.figure(1)
    fig.set_size_inches(10,10)
    plt.xlabel('Numero de epocas')
    plt.ylabel('EQM(Função de perda)')
    #plt.grid(True)
    plt.plot(range(0,len(vet_erro_tr)), vet_erro_tr, label='Treinamento %s' % j)
    plt.plot(range(0,len(vet_erro_val)), vet_erro_val, label='Validação %s' % j)
    plt.title('Evolução EQM')
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig(os.path.join("resultados","evolucao_EQM_%s.png" % nome), bbox_inches='tight')
