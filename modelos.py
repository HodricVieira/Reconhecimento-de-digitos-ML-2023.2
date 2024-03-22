import numpy as np
import matplotlib.pyplot as plt

class RegressaoLogistica():
    def __str__(self):
        return "Regressao Logistica"
    
    def __init__(self, eta = 0.000001, max_iter = 1000):
        self.eta = eta
        self.max_iteracoes = max_iter
    # Função que vai atribuir a cada amostra uma probabilidade de ser 1 ou 5 (1 ou -1)
    def probabilidades(self,w, X):
        z = np.matmul(w, X) # Esse aqui é o X.T para a operaçao ser possivel. Tambem pode ser feito com np.dot(w, X)
        probabilidades_amostrais = 1/(1+np.exp(-z))
        return probabilidades_amostrais
    
    # Funçção que vai atribuir a cada amostra uma classe de acordo com sua probabilidade
    def atribuicao_de_classes(self,probabilidades_amostrais):
        classes_amostrais_previstas = np.zeros(len(probabilidades_amostrais))
        classes_amostrais_previstas[np.where(probabilidades_amostrais<0.5)] = -1
        classes_amostrais_previstas[np.where(probabilidades_amostrais>=0.5)] = 1
        return classes_amostrais_previstas
    
    # Função que vai calcular a acuracia do modelo
    def calculate_accuracy(self,classes_amostrais_previstas, y):
        accuracy = (sum(classes_amostrais_previstas==y)/len(y))*100
        return accuracy


    def fit(self, X, Y):
        y = Y
        accuracies = []
        w = np.zeros(X.shape[0])
        m = len(y)
        
        for i in range(self.max_iteracoes):
            probabilidades_amostrais = self.probabilidades(w, X)
            classes_amostrais_previstas = self.atribuicao_de_classes(probabilidades_amostrais)
            accuracies.append(self.calculate_accuracy(classes_amostrais_previstas, y))
            erros = probabilidades_amostrais - y
            gradiente = (1/m)*np.matmul(X,erros)
            w = w - (self.eta*gradiente)
        
        self.classes_amostrais_previstas = classes_amostrais_previstas
        self.w = w
        self.accuracy = accuracies[-1]

        return w, accuracies[-1], classes_amostrais_previstas
        #return classes_amostrais_previstas

    def get_w(self):
        return self.w
    
    def plot_grafico(self, X, y_pred, valor1, valor2):
        # Criando a reta para plotar o gráfico
        x = np.linspace(-2, 2, 100)
        y_plot = (-self.w[0] - self.w[1]*x) / self.w[2]
        # printando as bolinhas vermelhas e azuis, se a classe for 1 (numero 1), plota azul, se for -1, plota vermelho
        plt.scatter(X[y_pred == 1, 1], X[y_pred == 1, 2], color='blue', marker='o', label=f'{valor1}')
        plt.scatter(X[y_pred == -1, 1], X[y_pred == -1, 2], color='red', marker='o', label=f'{valor2}')
        plt.plot(x, y_plot, label='Regressão Logística')
        plt.xlabel('Intensidade')
        plt.ylabel('Simetria')
        plt.title('Intensidade x Simetria')
        plt.legend()
        plt.show()

####################################################################################################
####################################################################################################

class RegressaoLinear():
    def __str__(self):
        return "Regressao Linear"
    
    def calculate_accuracy(self,classes_amostrais_previstas, y):
        accuracy = (sum(classes_amostrais_previstas==y)/len(y))*100
        return accuracy
    
    def fit(self, _X, _y):
        X = np.array(_X) # X é uma matriz de amostras
        Y = np.array(_y)
        XtX = np.dot(X.transpose(), X)
        XtX_inv = np.linalg.inv(XtX)
        XtY = np.dot(X.transpose(), Y)
        self.w = np.dot(XtX_inv, XtY) # w é um vetor unico
        return self.w
     
    def atribuicao_de_classes(self, _x):
        x = np.array(_x)
        classes_amostrais_previstas = []
        probabilidades = np.dot(x, self.w)
        for probabilidade in probabilidades:
            if probabilidade > 0:
                classes_amostrais_previstas.append(1)
            else:
                classes_amostrais_previstas.append(-1)
        
        return np.array(classes_amostrais_previstas)
     
    def getW(self):
        return self.w
    
    def plot_grafico(self, X, y_pred, valor1, valor2):
        # Criando a reta para plotar o gráfico
        x = np.linspace(-2, 2, 100)
        y_plot = (-self.w[0] - self.w[1]*x) / self.w[2]
        # printando as bolinhas vermelhas e azuis, se a classe for 1 (numero 1), plota azul, se for -1, plota vermelho
        plt.scatter(X[y_pred == 1, 1], X[y_pred == 1, 2], color='blue', marker='o', label=f'{valor1}')
        plt.scatter(X[y_pred == -1, 1], X[y_pred == -1, 2], color='red', marker='o', label=f'{valor2}')
        plt.plot(x, y_plot, label='Regressão Linear')
        plt.xlabel('Intensidade')
        plt.ylabel('Simetria')
        plt.title('Intensidade x Simetria')
        plt.legend()
        plt.show()

####################################################################################################
####################################################################################################

class PLA():
    def __str__(self):
        return "Perceptron Learning Algorithm"
    
    def __init__(self, max_iter= 100):
        self.max_iter = max_iter
        

    def fit(self, _X, _Y):
        X = np.array(_X)
        y = np.array(_Y)
        self.w = np.zeros(len(X[0]))
        bestError = len(y)
        bestW = self.w

        for iter in range(self.max_iter):         
            #Testa se sign(wTXn) != Yn - ponto classificado errado
            for i in range(len(y)):
                if(np.sign(np.dot(self.w, X[i])) != y[i]):
                    self.w = self.w + (y[i]*X[i])
                    eIN = self.errorIN(X, y)
                    if(bestError > eIN):
                        bestError = eIN
                        bestW = self.w

    def errorIN(self, X, y):
        error = 0
        for i in range(len(y)):
            if(np.sign(np.dot(self.w, X[i])) != y[i]):
                error += 1
        return error

    def getW(self):
        return self.w

    def atribuicao_de_classes(self, _x):
        x = np.array(_x)
        classes_amostrais_previstas = []
        probabilidades = np.dot(x, self.w)
        for probabilidade in probabilidades:
            if probabilidade > 0:
                classes_amostrais_previstas.append(1)
            else:
                classes_amostrais_previstas.append(-1)
        return np.array(classes_amostrais_previstas)

    def calculate_accuracy(self,classes_amostrais_previstas, y):
        accuracy = (sum(classes_amostrais_previstas==y)/len(y))*100
        return accuracy

    def plot_grafico(self, X, y_pred, valor1, valor2):
        # Criando a reta para plotar o gráfico
        x = np.linspace(-2, 2, 100)
        y_plot = (-self.w[0] - self.w[1]*x) / self.w[2]
        # printando as bolinhas vermelhas e azuis, se a classe for 1 (numero 1), plota azul, se for -1, plota vermelho
        plt.scatter(X[y_pred == 1, 1], X[y_pred == 1, 2], color='blue', marker='o', label=f'{valor1}')
        plt.scatter(X[y_pred == -1, 1], X[y_pred == -1, 2], color='red', marker='o', label=f'{valor2}')
        plt.plot(x, y_plot, label='Perceptron')
        plt.xlabel('Intensidade')
        plt.ylabel('Simetria')
        plt.title('Intensidade x Simetria')
        plt.legend()
        plt.show()