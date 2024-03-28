import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

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
        accuracy = (sum(classes_amostrais_previstas==y)/len(y))*100
        return accuracy


    def fit(self):
        num_iter = 0
        for i in range(self.max_iteracoes):
            probabilidades_amostrais = self.probabilidades(self.w, self.X)
            erros = probabilidades_amostrais - np.reshape(self.y, (len(self.y), 1))
            gradiente = np.dot(self.X.T,erros)
            self.w = self.w - (self.eta*gradiente)
            num_iter += 1
            if LA.norm(gradiente) < 0.0001 :
                break

    def get_w(self):
        return self.w
    
    def plot_grafico(self,X, y_pred, valor1, valor2):
        plt.scatter(X[y_pred == 1, 1], X[y_pred == 1, 2], color='blue', marker='o', label=f'{valor1}')
        plt.scatter(X[y_pred == -1, 1], X[y_pred == -1, 2], color='red', marker='o', label=f'{valor2}')
        x_min = np.min(X[:, 1]) - 0.5 # Menor valor da coluna de intensidade (eixo x)
        x_max = np.max(X[:, 1]) + 0.5 # Maior valor da coluna de intensidade (eixo x)
        y_min = np.min(X[:, 2]) - 0.5 # Menor valor da coluna de simetria (eixo y)
        y_max = np.max(X[:, 2]) + 0.5 # Maior valor da coluna de simetria (eixo y)
        x = np.linspace(x_min, x_max, 100)
        y_plot = (-self.w[0] - self.w[1]*x) / self.w[2]
        plt.plot(x, y_plot, label="Regressao Logistica")
        plt.title(f"Regressão Logística - Intensidade x Simetria")
        plt.xlabel("Intensidade")
        plt.ylabel("Simetria")
        plt.legend()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)        
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
    
    def plot_grafico(self,X, y_pred, valor1, valor2):
        plt.scatter(X[y_pred == 1, 1], X[y_pred == 1, 2], color='blue', marker='o', label=f'{valor1}')
        plt.scatter(X[y_pred == -1, 1], X[y_pred == -1, 2], color='red', marker='o', label=f'{valor2}')
        x_min = np.min(X[:, 1]) - 0.5 # Menor valor da coluna de intensidade (eixo x)
        x_max = np.max(X[:, 1]) + 0.5 # Maior valor da coluna de intensidade (eixo x)
        y_min = np.min(X[:, 2]) - 0.5 # Menor valor da coluna de simetria (eixo y)
        y_max = np.max(X[:, 2]) + 0.5 # Maior valor da coluna de simetria (eixo y)
        x = np.linspace(x_min, x_max, 100)
        y_plot = (-self.w[0] - self.w[1]*x) / self.w[2]
        plt.plot(x, y_plot, label="Regressao Linear")
        plt.title(f"Regressão Linear - Intensidade x Simetria")
        plt.xlabel("Intensidade")
        plt.ylabel("Simetria")
        plt.legend()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)        
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

    def plot_grafico(self,X, y_pred, valor1, valor2):
        plt.scatter(X[y_pred == 1, 1], X[y_pred == 1, 2], color='blue', marker='o', label=f'{valor1}')
        plt.scatter(X[y_pred == -1, 1], X[y_pred == -1, 2], color='red', marker='o', label=f'{valor2}')
        x_min = np.min(X[:, 1]) - 0.5 # Menor valor da coluna de intensidade (eixo x)
        x_max = np.max(X[:, 1]) + 0.5 # Maior valor da coluna de intensidade (eixo x)
        y_min = np.min(X[:, 2]) - 0.5 # Menor valor da coluna de simetria (eixo y)
        y_max = np.max(X[:, 2]) + 0.5 # Maior valor da coluna de simetria (eixo y)
        x = np.linspace(x_min, x_max, 100)
        y_plot = (-self.w[0] - self.w[1]*x) / self.w[2]
        plt.plot(x, y_plot, label="Perceptron")
        plt.title(f"PLA - Intensidade x Simetria")
        plt.xlabel("Intensidade")
        plt.ylabel("Simetria")
        plt.legend()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)        
        plt.show()