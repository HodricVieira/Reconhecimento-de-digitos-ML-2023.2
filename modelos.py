import numpy as np

class RegressaoLogistica():
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

    def get_w(self):
        return self.w

####################################################################################################
####################################################################################################

class RegressaoLinear():
    def fit(self, _X, _y):
        X = np.array(_X) # X é uma matriz de amostras
        Y = np.array(_y)
        XtX = np.dot(X.transpose(), X)
        XtX_inv = np.linalg.inv(XtX)
        XtY = np.dot(X.transpose(), Y)
        self.w = np.dot(XtX_inv, XtY) # w é um vetor unico
     
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

####################################################################################################
####################################################################################################
