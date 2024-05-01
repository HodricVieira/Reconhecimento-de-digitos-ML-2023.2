import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import modelos as md

# Função que vai padronizar a matriz X para uma normal 0,1
def padronizar_normal(matriz_X):
    for i in range(np.shape(matriz_X)[1]):
        matriz_X[:,i] = (matriz_X[:,i] - np.mean(matriz_X[:,i]))/np.std(matriz_X[:,i]) # X menos media/desvio padrao

    return matriz_X
############################################################################################################################################################################
############################################################################################################################################################################

# Função que vai plotar a matriz de confusão
def ver_confusao(matriz_de_confusao):
    classes = ['1', '5']
    plt.matshow(matriz_de_confusao, cmap='Blues')
    plt.title('Matriz de confusão')
    plt.colorbar()
    # Adicionando rótulos aos eixos
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.ylabel("Classificações corretas")
    plt.xlabel("Classificações obtidas")
    plt.show()
############################################################################################################################################################################
############################################################################################################################################################################


# Função que vai retornar o relatório de classificação do treino
def ver_treino(X, Y, a, b, modelo):
    # a e b são os dois indices escolhidos
    y_pred_treino = modelo.atribuicao_de_classes(X)

    #Criando a matriz de confusão
    matriz_de_confusao_treino = confusion_matrix(Y, y_pred_treino)
    print(matriz_de_confusao_treino)

    #Printando o relatório de classificação
    print(classification_report(Y, y_pred_treino, target_names=[str(b), str(a)]))

    # Criando a reta para plotar o gráfico
    modelo.plot_grafico(X, y_pred_treino, a, b)
    ver_confusao(matriz_de_confusao_treino)


# Função que vai retornar o relatório de classificação do teste
def ver_teste(X_teste, y_teste, a, b, modelo):
    y_pred_test = modelo.atribuicao_de_classes(X_teste)

    #Printando a matriz de confusao
    matriz_de_confusao_teste = confusion_matrix(y_teste, y_pred_test)
    print(matriz_de_confusao_teste)

    #Printando o relatório de classificação
    print(classification_report(y_teste, y_pred_test, target_names=[str(b), str(a)]))

    modelo.plot_grafico(X_teste, y_pred_test, a, b)
    ver_confusao(matriz_de_confusao_teste)
############################################################################################################################################################################
############################################################################################################################################################################

# Função responsável pela classificação 1 vs todos da regressão logística
def classificador_de_todos_os_digitos_Rlog(X_treino,Y_treino, X_teste, Y_teste, lista_digitos):
    X = X_treino.copy()
    x_teste = X_teste.copy()
    y = Y_treino.copy()
    y_teste = Y_teste.copy()

    for digito in lista_digitos:     
        y_binario_treino = np.where(y == digito, 1, 0) # Com 1 e 0 o resultado do fit é mais preciso que com 1 e -1
        y_binario_teste = np.where(y_teste == digito, 1, -1)
        modelo_reg_log = md.RegressaoLogistica(X,y_binario_treino)
        modelo_reg_log.fit()

        # Classificação do treino
        y_pred_treino = modelo_reg_log.atribuicao_de_classes(modelo_reg_log.get_w(),X) # y_pred_treino retorna 1 e -1 (como especificado no projeto)
        # Classificação do teste
        y_pred_teste = modelo_reg_log.atribuicao_de_classes(modelo_reg_log.get_w(),x_teste)
        # Alterando o y_binario_treino (onde tiver 0, troco para -1, pois o y_pred_treino é 1 ou -1)
        y_binario_treino = np.where(y_binario_treino == 0, -1, 1)

        # Relatorio do treino
        print('Digito:', digito)
        print('Acuracia do treino:', modelo_reg_log.calculate_accuracy(y_pred_treino, y_binario_treino))
        print('Matriz de confusão do treino, os valores do digito estao na linha de baixo')
        matriz_de_confusao_treino = confusion_matrix(y_binario_treino, y_pred_treino)
        print(matriz_de_confusao_treino)
        print(classification_report(y_binario_treino, y_pred_treino, target_names=['-1',f'{digito}']))
        # Gráfico do treino
        print('Gráfico de treino')
        modelo_reg_log.plot_grafico(X, y_pred_treino, digito, -1)

        # Relatorio do teste
        print('Digito:', digito)
        print('Acuracia do teste:', modelo_reg_log.calculate_accuracy(y_pred_teste, y_binario_teste))
        print('Matriz de confusão do teste, os valores do digito estao na linha de baixo')
        matriz_de_confusao_teste = confusion_matrix(y_binario_teste, y_pred_teste)
        print(matriz_de_confusao_teste)
        print(classification_report(y_binario_teste, y_pred_teste, target_names=['-1',f'{digito}']))
        # Gráfico do teste
        print('Gráfico de teste')
        modelo_reg_log.plot_grafico(x_teste, y_pred_teste, digito, -1)

        # obtenha o indice dos valores de y_pred_treino que são 1 e retirando os elementos correspondentes em X e Y
        indices = np.where(y_pred_treino == 1)[0]
        X = np.delete(X, indices, axis=0)
        y = np.delete(y, indices, axis=0)
        indices_teste = np.where(y_pred_teste == 1)[0]
        x_teste = np.delete(x_teste, indices_teste, axis=0)
        y_teste = np.delete(y_teste, indices_teste, axis=0)
    
############################################################################################################################################################################
############################################################################################################################################################################

# Função responsável pela classificação 1 vs todos da regressão linear e do PLA
def classificador_de_todos_os_digitos(X_treino,Y_treino,  X_teste, Y_teste, lista_digitos, modelo_reg):
    X = X_treino.copy()
    y = Y_treino.copy()
    x_teste = X_teste.copy()
    y_teste = Y_teste.copy()

    for digito in lista_digitos:     
        y_binario_treino = np.where(y == digito, 1, -1)
        y_binario_teste = np.where(y_teste == digito, 1, -1)

        #treinando o modelo com os dados de treino
        w = modelo_reg.fit(X, y_binario_treino)
        y_pred_treino = modelo_reg.atribuicao_de_classes(X) 
        #utilizando os dados de teste com o w encontrado no treino
        y_pred_teste = modelo_reg.atribuicao_de_classes(x_teste)

        # Relatorio do treino
        print('Digito:', digito)
        print('Acuracia do treino:', modelo_reg.calculate_accuracy(y_pred_treino, y_binario_treino))
        print('Matriz de confusão do treino, os valores do digito estao na linha de baixo')
        matriz_de_confusao_treino = confusion_matrix(y_binario_treino, y_pred_treino)
        print(matriz_de_confusao_treino)
        print(classification_report(y_binario_treino, y_pred_treino, target_names=['-1',f'{digito}']))
        # Gráfico do treino
        print('Gráfico de treino')
        modelo_reg.plot_grafico(X, y_pred_treino, digito, -1)

        # Relatório do teste
        print('Digito:', digito)
        print('Acuracia do teste:', modelo_reg.calculate_accuracy(y_pred_teste, y_binario_teste))
        matriz_de_confusao_teste = confusion_matrix(y_binario_teste, y_pred_teste)
        print(matriz_de_confusao_teste)
        print(classification_report(y_binario_teste, y_pred_teste, target_names=['-1',f'{digito}']))
        print('Gráfico de teste')
        # plotando o gráfico de teste
        modelo_reg.plot_grafico(x_teste, y_pred_teste, digito, -1)
        

        # obtenha o indice dos valores de y_pred_treino que são 1 e retirando os elementos correspondentes em X e Y
        indices = np.where(y_pred_treino == 1)[0]
        X = np.delete(X, indices, axis=0)
        y = np.delete(y, indices, axis=0)
        indices_teste = np.where(y_pred_teste == 1)[0]
        x_teste = np.delete(x_teste, indices_teste, axis=0)
        y_teste = np.delete(y_teste, indices_teste, axis=0)


