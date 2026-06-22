from utils import armar_matriz
from sklearn.preprocessing import normalize
import numpy as np
#Esta funcion simula k matrices de m vertices con igual soporte.
def simular(m,k):
    l = (m*(m-1)//2)
    b = 0.2
    mu = 3 #la media de los pesos de los aristas y su 
    sigma = 4 # varianza
    soporte = np.random.binomial(1,b,size= l)
    while fila_suma_cero(soporte,m):
        soporte = np.random.binomial(1,b,size= l)
    pesos = np.zeros((l,k))
    for i in range(0,k):
        pesos_i = np.random.normal(mu,sigma,size=l) 
        pesos[:,i] = np.abs(pesos_i)*soporte
    

        
    matrices_aplastadas = pesos
    
    
    return [matrices_aplastadas[:,i]  for i in range(0,k)]
#funcion auxiliar a la anterior:
def fila_suma_cero(W1_flat,m):
    #para cierta simulacion devuelve False si ninguna fila suma cero y True sino
    W1_mat = armar_matriz(W1_flat,m)
    return (np.sum(W1_mat,axis=1)==np.zeros((m,m))).any() 




####################################################################################3
#Dada una mariz de m nodos generar una señal de largo m.
def senal(W_1,m,n):
    W_1 = armar_matriz(W_1,m)
    D_1 = np.diag(sum(W_1))
    
    L_1 = D_1 - W_1 
    
    L_1_pinv = np.linalg.pinv(L_1)
    
    
    X_1 = []
    for i in range(n):
        muestra_señal = np.random.multivariate_normal(np.zeros(m),L_1_pinv)
        X_1.append(muestra_señal) 
    X_1 = np.array(X_1)
    X_1 = normalize(X_1)
    
    return X_1
