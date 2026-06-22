import numpy as np
from sklearn.preprocessing import normalize
####################################################################################################

#Construccion de la matriz auxiliar que se usa para definir el termino que involucra al logaritmo.
def crear_matriz_aux(m):
    vector = np.arange(m)
    parcial_sums = [sum(vector[i:]) for i in range(1,m+1)]
    M = []
    for i in range(1,m):
        fila_i = np.zeros(int(m*(m-1)/2))
        inicio = parcial_sums[-i]
        fin = parcial_sums[-i-1]
        fila_i[inicio:fin] = list(np.ones(m-i)) 
        M.append(list(fila_i))
    fila_i = np.zeros(int(m*(m-1)/2))
    M.append(list(fila_i))
    M  = np.array(M).reshape(m,int(m*(m-1)/2))
    
    vector_2 = np.arange(m-1)
    parcial_sums_2 = [sum(vector_2[i:]) for i in range(1,m)]
    parcial_sums_2
    
    N=[]
    for fila in range(1,m):
        lista_fila=[]
        for j in range(fila):
            lista_fila.append(parcial_sums_2[m-j-2]+fila-1)
        N.append(lista_fila)
    for i,fila in enumerate(N):
        for indice in fila:
            M[i+1,indice] = 1
    return M



# Creamos un diccionario para guardar M. Si m no cambia, la usamos de acá
# en lugar de volver a construirla en cada iteración del gradiente y la función.
_CACHE_M = {}

def _obtener_M(m_nodos):
    if m_nodos not in _CACHE_M:
        _CACHE_M[m_nodos] = crear_matriz_aux(m_nodos)
    return _CACHE_M[m_nodos]

def distancias(X,m):
    X = np.array(X)
    X = normalize(X)
    
    n,m = X.shape
    Z = np.zeros((m,m))  
    for i in range(m):
        for j in range(m):
            sign_i = np.array([X[k][i] for k in range(0,n)])
            sign_j = np.array([X[k][j] for k in range(0,n)])
            dist = np.linalg.norm(sign_i-sign_j)
            Z[i][j] = dist
    
    
    
    Z = Z[np.triu_indices(m, k = 1)] 

    return Z

##################################################################################
def matriz_binaria(W_flat):
    #funcion auxiliar de error_sop
    #devuelve una matriz con unos en el soporte y los ceros los deja igual.
    W = W_flat.copy()
    W[W > 0] = 1
    return W

def armar_matriz(W,m):
    X = np.zeros((m,m))
    X[np.triu_indices(m, k = 1)] = W
    return X + X.T