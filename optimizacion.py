import numpy as np
from utils import _obtener_M
##############################################################################################
######################********************####################################################
#####################* Kalafolias     *#####################################################
####################**********************####################################################
def kala(W, Z, alpha, beta): #come el triangulo superior aplastado
    # np.dot(W, W) es equivalente a np.linalg.norm(W)**2 pero MUCHO más rápido
    return 2 * np.dot(W, Z) - alpha * term_log(W) + 2 * beta * np.dot(W, W)


#####################################
#FUNCION AUXILIAR A LA ANTERIOR
def term_log(W_a): #come el triangulo superior aplastado
    l = len(W_a)
    m_nodos = int((1 + np.sqrt(8 * l + 1)) / 2)
    
    M = _obtener_M(m_nodos) # <--- Buscamos M en el caché
    
    v = M @ W_a
    # sumamos epsilon (1e-12) para evitar warnings matemáticos si algún valor es 0
    return np.sum(np.log(np.maximum(v, 1e-12))) 


############################################################################################

def grad_kala(W_flat, Z_flat, alpha, beta):
    return 2 * Z_flat - alpha * grad_term_log(W_flat) + 4 * beta * W_flat
    
#####################################
#FUNCION AUXILIAR A LA ANTERIOR
def grad_term_log(W_flat):
    l = len(W_flat)
    m_nodos = int((1 + np.sqrt(8 * l + 1)) / 2)
    
    M = _obtener_M(m_nodos) # <--- Buscamos M en el caché
    
    v = M @ W_flat
    
    # 🚀 VECTORIZACIÓN TOTAL:
    # 1. División matricial nativa en C (sin bucles 'for' de Python)
    v_inv = 1.0 / np.maximum(v, 1e-12) 
    
    # 2. Reemplazamos el bucle 'for j, fila in enumerate(M.T)' 
    # por una simple multiplicación matriz-vector. Esto vuela.
    grad = M.T @ v_inv 
    
    return grad



##############################################################################
def f(W_flat, alpha, beta):
    return -alpha * term_log(W_flat) + 2 * beta * np.dot(W_flat, W_flat)

def grad_f(W_flat, alpha, beta):
    return -alpha * grad_term_log(W_flat) + 4 * beta * W_flat



######################********************####################################################
#####################*Funciones proximales*###################################################
####################**********************####################################################

def prox_g(W_flat, Z_flat, lambd): #todas aplastadas
    return np.sign(W_flat) * np.maximum(np.abs(W_flat) - lambd * 2 * Z_flat, 0)

#############################################################################################
def algoritmo_proximal(W_init, Z_flat, alpha, beta, lambd, iters):
    W_act = W_init.copy()
    valores_kala = []
    
    # Precalculamos esta constante para no multiplicarla 'iters' veces adentro del proximal
    lambd_2_Z = lambd * 2 * Z_flat 
    
    for i in range(iters):
        W_desc = W_act - lambd * grad_f(W_act, alpha, beta)
        
        # Invocamos la lógica del proximal directo acá para ahorrar el llamado a la función
        W_act = np.sign(W_desc) * np.maximum(np.abs(W_desc) - lambd_2_Z, 0)

        valores_kala.append(kala(W_act, Z_flat, alpha, beta))

    return W_act, valores_kala
############################********************#############################################
##########################* Infernecia conjunta *############################################
###########################*********************#############################################
epsilon = 0.0001 #condicion de parada de lasso, tener diferencia de la norma entre W y T(W) menor que epsilon.
def term_grouped_lasso(lista_matrices):
    # Vectorización total: apila en columnas y calcula normas por filas nativamente
    W_flat = np.column_stack(lista_matrices)
    normas = np.linalg.norm(W_flat, axis=1)
    return np.sum(normas)

def grouped_lasso(lista_matrices, lista_distancias, alpha, beta, a):
    k = len(lista_matrices)
    sum_kala = sum(kala(lista_matrices[i], lista_distancias[i], alpha, beta) for i in range(k))
    return sum_kala + a * term_grouped_lasso(lista_matrices)

###########################################################################################
# proximal method con la funcion grouped lasso

def threshold(w, a, k):
    # Calcular la norma UNA sola vez
    norm_w = np.linalg.norm(w)
    if norm_w == 0:
        return np.zeros(k)
    return w * np.maximum(norm_w - a, 0) / norm_w

def prox_h(lista_matrices, lista_distancias, lambd, a):
    k = len(lista_matrices)
    
    # column_stack es el equivalente rápido a armar un W_flat con reshapes concatenados
    W_flat = np.column_stack(lista_matrices)
    Z_flat = np.column_stack(lista_distancias)
    
    W_tilde = prox_g(W_flat, Z_flat, lambd)

    return [lambd * threshold(w / lambd, a, k) for w in W_tilde]

##############################################################################################

def elgir_paso_armijo(lista_matrices, grad_actual, f_actual, lambd, c, tao, alpha, beta):
    # Recibe los gradientes y valores de f ya precalculados para no recalcularlos
    normas_grad_cuadrado = [np.linalg.norm(g)**2 for g in grad_actual]
    
    arm = 0
    while arm < 10:
        cumple_todas = True
        for i, w in enumerate(lista_matrices):
            w_new = w - lambd * grad_actual[i]
            # Condición de Armijo: si f(nuevo) >= f(viejo) - lambda * norma^2 * c, falló
            if f(w_new, alpha, beta) >= f_actual[i] - lambd * normas_grad_cuadrado[i] * c:
                cumple_todas = False
                break # NO evaluamos el resto de las matrices, ahorramos tiempo
        
        if cumple_todas:
            break
            
        lambd *= tao
        arm += 1
        
    return lambd

def algoritmo_proximal_lasso(lista_matrices_init, lista_distancias, alpha, beta, lambd, a, iters_max, armijo=True, c=0.1, tao=0.1, iter_reporte=[],epsilon = 0.0001):
    lista_actualizadas = [W.copy() for W in lista_matrices_init]
    
    k = len(lista_matrices_init)
    valores_lasso = []
    i = 0
    condicion_siga = True
    
    while (condicion_siga & (i<iters_max)):
        # 1. Precalcular gradientes UNA VEZ por iteración (sirve para Armijo y Descenso)
        grad_actual = [grad_f(W, alpha, beta) for W in lista_actualizadas]
        
        # 2. Paso Armijo
        if armijo and i > 0:
            # Precalcular f_actual para pasarlo a Armijo
            f_actual = [f(W, alpha, beta) for W in lista_actualizadas]
            lambd = elgir_paso_armijo(lista_actualizadas, grad_actual, f_actual, lambd, c, tao, alpha, beta)
            #if i in iter_reporte: # Opcional: imprimir lambd solo si reporta
                 #print(f"Lambd elegido: {lambd}")
        #guardar la ultima actualizacion antes de sobreescribirla
        lista_anteriores = lista_actualizadas
        # 3. Descenso usando el gradiente que ya calculamos arriba
        lista_descendidas = [W - lambd * g for W, g in zip(lista_actualizadas, grad_actual)]
        
        # 4. Proximal
        lista_prox_h = prox_h(lista_descendidas, lista_distancias, lambd, a)
        
        # 5. Extraer columnas convirtiendo a array UNA sola vez
        arr_prox = np.array(lista_prox_h)
        
        lista_actualizadas = [arr_prox[:, j] for j in range(k)]
        
        valores_lasso.append(grouped_lasso(lista_actualizadas, lista_distancias, alpha, beta, a))
        i = i+1
        #if i in iter_reporte:
            #print(f"Corrió iteración {i}")
        condicion_siga = np.linalg.norm(np.array(lista_actualizadas)-np.array(lista_anteriores))  > epsilon
    return np.array(lista_actualizadas), valores_lasso