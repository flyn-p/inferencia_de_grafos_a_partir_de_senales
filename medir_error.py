import numpy as np

def error_norm(W_inferida,W_original):
    #esta funcion implementa el error de la distancia de la norma, con una normalización para que las matrices sean comparables.
    norm_ori = np.linalg.norm(W_original)
    norm_infe = np.linalg.norm(W_inferida)
    return np.linalg.norm(W_inferida/norm_infe- W_original/norm_ori)

def matriz_binaria(W_flat):
    #funcion auxiliar de error_sop
    #devuelve una matriz con unos en el soporte y los ceros los deja igual.
    W = W_flat.copy()
    W[W > 0] = 1
    return W

def error_sop(W_inferida,W_original):
    #esta funcion implementa el error de la diferencia simétrica del soporte.
    W_inf_bin = matriz_binaria(W_inferida)
    W_ori_bin = matriz_binaria(W_original)
    return np.sum(np.abs(W_inf_bin -W_ori_bin))

