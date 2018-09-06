# -*- coding: utf-8 -*-

import numpy as np
import math
from mnist import MNIST
import time
import warnings
warnings.filterwarnings('ignore')

def load_images():
    mndata = MNIST()
    
    training_images, training_labels = mndata.load_training()
    test_images    , test_labels     = mndata.load_testing()

    return training_images, training_labels, test_images, test_labels

#Função que faz a regressão linear e compara os resultados com as imagens de teste
def linear_regression(training_images, training_labels, test_images, test_labels):

    #Calcula a pseudo-inversa e depois calcula w de acordo com os slides do Yaser
    pseudoInverse = np.linalg.pinv(training_images)
    w = np.matmul(pseudoInverse, training_labels)

    counter = 0
    #Compara com as imagens de teste e conta os acertos
    for i in range(0, len(test_images)):
        num = abs(round(np.dot(w, test_images[i])))
        if(num == test_labels[i]):
            counter = counter + 1

    return counter

#Função que conta quantidade de pixels brancos e não-brancos na imagem
def whites_and_blacks(images):

    result_images = []

    for i in range(0, len(images)):
        whites = 0
        blacks = 0
        for j in range(0, len(images[i])):
            if(images[i][j] == 0):
                whites = whites + 1

        blacks    = j - whites + 1
        result_images.append(np.array([blacks, whites]))

    return result_images

#Função que extrai as diversas métricas das imagens
def extract_info(images):

    result_images = []

    for i in range(0, len(images)):
        whites = 0
        blacks = 0

        #Extrai quantidade de pixels brancos e não-brancos
        for j in range(0, len(images[i])):
            if(images[i][j] == 0):
                whites = whites + 1

        blacks = j - whites + 1

        #Extrai intensidade média e desvio padrão da intensidade
        avg_intensity = np.average(images[i])
        std_deviation = np.std(images[i])
        
        #Transforma o array em matriz para extrair autovalores
        matrix = np.zeros((28,28))
        for k in range(0, 28):
            for l in range(0,28):
                matrix[k][l] = images[i][k * 28 + l]
        matrix = np.cov(matrix)
        eigenvalue, eigenvector = np.linalg.eig(matrix)
        
        #Maior, menor e valor médio dos autovalores
        max_eigenvalue  = np.max(eigenvalue)
        min_eigenvalue  = np.min(eigenvalue)
        avg_eigenvalue  = np.average(eigenvalue)
        
        #Utilizando todas as métricas
        result_images.append(np.array([whites, blacks, avg_intensity, std_deviation, max_eigenvalue, min_eigenvalue, avg_eigenvalue]))
        
        #Utilizando apenas quantidade de pixels não-brancos, intensidade média e maior autovalor
        #result_images.append(np.array([blacks, avg_intensity, max_eigenvalue]))
    return result_images


start = time.time()
training_images, training_labels, test_images, test_labels = load_images()
end = time.time()

print "Tempo para carregar as imagens: " + str(end - start) + "s"

#Primeira vez tenta com os valores puros de cada pixel da imagem
start = time.time()
acertos = linear_regression(training_images, training_labels, test_images, test_labels)
end = time.time()

print "Taxa de acerto raw input: " + str((float(acertos)/len(test_images)) * 100) + "%"
print "Tempo para treinar + testar raw input: " + str(end - start) + "s"

#Faz a classificação utilizando a quantidade de pixels brancos e não-brancos
start = time.time()

bw_training_images = whites_and_blacks(training_images)
bw_test_images     = whites_and_blacks(test_images)
acertos         = linear_regression(bw_training_images, training_labels, bw_test_images, test_labels)

end = time.time()
print "Taxa de acerto contando pixels brancos e pretos: " + str((float(acertos)/len(test_images)) * 100) + "%"

print "Tempo para treinar + testar pixels brancos e pretos: " + str(end - start) + "s"

#Faz a classificação utilizando diversas métricas
start = time.time()

training_images = extract_info(training_images)
test_images     = extract_info(test_images)
acertos         = linear_regression(training_images, training_labels, test_images, test_labels)

end = time.time()
print "Taxa de acerto métricas diversas: " + str((float(acertos)/len(test_images)) * 100) + "%"

print "Tempo para treinar + testar métricas diversas: " + str(end - start) + "s"