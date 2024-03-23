import numpy as np
import matplotlib.pyplot as plt

class RegressaoLogistica():
    def __str__(self):
        return "Regressao Logistica"
    
    def __init__(self,X_, y_, eta = 0.0001, max_iter = 200):
        self.eta = eta
        self.max_iteracoes = max_iter
        self.y = np.array(y_)
        self.X = np.array(X_)
        self.w = np.zeros((self.X.shape[1], 1))
    # Função que vai atribuir a cada amostra uma probabilidade de ser 1 ou 5 (1 ou -1)
    def probabilidades(self,w, X):
        z = np.dot(X, w)
        return 1 / (1 + np.exp(-z))
    
    # Funçção que vai atribuir a cada amostra uma classe de acordo com sua probabilidade
    def atribuicao_de_classes(self,w, x):
        classes_amostrais_previstas = np.array([])
        probabilidades_amostrais = self.probabilidades(w, x)
        for probabilidade in probabilidades_amostrais:
            if probabilidade > 0.5:
                classes_amostrais_previstas = np.append(classes_amostrais_previstas, 1)
            else:
                classes_amostrais_previstas = np.append(classes_amostrais_previstas, -1)
        return classes_amostrais_previstas
    
    # Função que vai calcular a acuracia do modelo
    def calculate_accuracy(self,classes_amostrais_previstas, y):
        # onde tiver -1, vai ser 0, onde tiver 1, vai ser 1
        classes_amostrais_previstas = np.where(classes_amostrais_previstas == -1, 0, 1)
        accuracy = (sum(classes_amostrais_previstas==y)/len(y))*100
        return accuracy


    def fit(self):
        for i in range(self.max_iteracoes):
            probabilidades_amostrais = self.probabilidades(self.w, self.X)
            erros = probabilidades_amostrais - np.reshape(self.y, (len(self.y), 1))
            gradiente = np.dot(self.X.T,erros)
            self.w = self.w - (self.eta*gradiente)

    def get_w(self):
        return self.w
    
    def plot_grafico(self,X, y_pred, valor1, valor2):
        plt.scatter(X[y_pred == 1, 1], X[y_pred == 1, 2], color='blue', marker='o', label=f'{valor1}')
        plt.scatter(X[y_pred == -1, 1], X[y_pred == -1, 2], color='red', marker='o', label=f'{valor2}')
        xmin = np.min(self.X[:, 1]) - 0.5
        xmax = np.max(self.X[:, 1]) + 0.5
        x = np.linspace(xmin, xmax, 100)
        y_plot = (-self.w[0] - self.w[1]*x) / self.w[2]
        plt.plot(x, y_plot, label="Regressao Logistica")
        plt.title(f"Regressão Logística - Intensidade x Simetria")
        plt.xlabel("Intensidade")
        plt.ylabel("Simetria")
        plt.legend()
        # limita com o maior e menor valor de x e y
        plt.xlim(xmin, xmax)
        plt.ylim(np.min(self.X[:, 2]) - 0.5, np.max(self.X[:, 2]) + 0.5)       
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
        plt.title('Regressao Linear - Intensidade x Simetria')
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
        plt.title('PLA - Intensidade x Simetria')
        plt.legend()
        plt.show()