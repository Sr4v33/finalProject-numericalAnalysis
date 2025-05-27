# core/chapterTwo/methods.py
import numpy as np

# Funciones auxiliares del nuevo código
def calculate_spectral_radius(T):
    try:
        if T is None or T.size == 0:
            return np.inf
        eigenvalues = np.linalg.eigvals(T)
        return np.max(np.abs(eigenvalues))
    except np.linalg.LinAlgError:
        return np.inf
    except Exception:
        return np.inf

def get_D_L_U(A): # Definición estándar A = D - L - U
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1) # Estrictamente triangular inferior
    U = -np.triu(A, 1)  # Estrictamente triangular superior
    return D, L, U

# El nuevo método unificado (tal como lo proporcionaste)
def iterative_method_core(A_in, b_in, x0_in, tol, nmax, method_type='jacobi', w=1.0):
    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)
    x0 = np.array(x0_in, dtype=float)

    # Diccionario de resultados que esta función llenará
    results_dict = {
        'D': None, 'L': None, 'U': None, 'T': None, 'C': None,
        'spectral_radius': np.inf, # Default a inf
        'iterations': [], # Lista de listas: [k, error, x_vector_list]
        'conclusion': "El método no se ejecutó completamente.",
        'final_solution': None,
        'convergence_message_detail': "No determinado." # Para el mensaje de ρ(T)
    }

    # Validaciones iniciales
    if np.any(np.diag(A) == 0) and method_type != 'sor': # SOR lo maneja diferente con D-wL
        if method_type == 'jacobi' or method_type == 'gauss_seidel':
             results_dict['conclusion'] = "Fracasó: La matriz tiene ceros en su diagonal principal (requerido para D o D-L invertible)."
             results_dict['convergence_message_detail'] = "No converge: Elemento diagonal cero."
             return results_dict
    
    try:
        if abs(np.linalg.det(A)) < 1e-12: # Chequeo de determinante (cercano a cero)
            results_dict['conclusion'] = "Fracasó: det(A) es (o muy cercano a) 0. La matriz es singular o casi singular."
            results_dict['convergence_message_detail'] = "No converge: Matriz singular."
            return results_dict
    except np.linalg.LinAlgError:
        results_dict['conclusion'] = "Fracasó: No se pudo calcular det(A) (posiblemente no cuadrada)."
        results_dict['convergence_message_detail'] = "Error de matriz."
        return results_dict

    if tol < 0 or nmax < 0:
        results_dict['conclusion'] = "Fracasó: Tolerancia o Nmax inválidos (deben ser positivos)."
        return results_dict
    
    if method_type == 'sor' and not (0 < w < 2):
        results_dict['conclusion'] = f"Fracasó: Omega (ω) para SOR debe estar en (0, 2), se recibió ω={w}."
        results_dict['convergence_message_detail'] = "Omega fuera de rango."
        return results_dict

    D, L, U = get_D_L_U(A)
    results_dict['D'], results_dict['L'], results_dict['U'] = D.tolist(), L.tolist(), U.tolist()

    T = None
    C_vec = None # C vector, no la matriz C del JS

    try:
        if method_type == 'jacobi':
            D_inv = np.linalg.inv(D)
            T = np.dot(D_inv, L + U)
            C_vec = np.dot(D_inv, b)
        elif method_type == 'gauss_seidel':
            DL_inv = np.linalg.inv(D - L)
            T = np.dot(DL_inv, U)
            C_vec = np.dot(DL_inv, b)
        elif method_type == 'sor':
            # (D - wL)x_k+1 = ( (1-w)D + wU )x_k + wb
            D_minus_wL = D - w * L
            D_minus_wL_inv = np.linalg.inv(D_minus_wL)
            T = np.dot(D_minus_wL_inv, ((1 - w) * D + w * U))
            C_vec = w * np.dot(D_minus_wL_inv, b) # Corregido C para SOR
        else:
            results_dict['conclusion'] = f"Fracasó: método no reconocido '{method_type}'."
            return results_dict
    except np.linalg.LinAlgError as e:
        results_dict['conclusion'] = f"Fracasó: Matriz singular durante cálculo de T y C ({e})."
        results_dict['convergence_message_detail'] = f"Error de álgebra lineal: {e}"
        return results_dict

    results_dict['T'] = T.tolist() if T is not None else None
    results_dict['C_vector_calc'] = C_vec.tolist() if C_vec is not None else None # Renombrado para evitar confusión con 'C' de la entrada JS
    
    current_spectral_radius = calculate_spectral_radius(T)
    results_dict['spectral_radius'] = current_spectral_radius

    if current_spectral_radius == np.inf:
        results_dict['convergence_message_detail'] = "No se pudo calcular el radio espectral (posiblemente T no válida)."
    elif current_spectral_radius >= 1:
        results_dict['convergence_message_detail'] = f"ADVERTENCIA: Radio espectral ρ(T) = {current_spectral_radius:.7f} ≥ 1. El método puede no converger o hacerlo lentamente."
        # No retornamos inmediatamente, permitimos que intente iterar.
    else:
        results_dict['convergence_message_detail'] = f"Converge teóricamente (ρ(T) = {current_spectral_radius:.7f} < 1)."


    x_prev = x0.copy()
    # Iteración 0: error no aplica, se muestra el x0
    results_dict['iterations'].append([0, np.nan, x_prev.copy().tolist()])

    error_norm = tol + 1
    k = 0

    while error_norm > tol and k < nmax:
        if T is None or C_vec is None: # Chequeo por si T y C no se pudieron calcular
            results_dict['conclusion'] = "Fracasó: Matrices de iteración T o C no pudieron ser calculadas."
            break
        
        x_current = np.dot(T, x_prev) + C_vec
        
        # Error absoluto ||x_k - x_k-1||_inf
        error_norm = np.max(np.abs(x_current - x_prev))
        
        k += 1
        results_dict['iterations'].append([k, error_norm, x_current.tolist()])
        x_prev = x_current.copy()

    results_dict['final_solution'] = x_prev.tolist() # El último x_prev es la solución x_k
    
    final_x_str = ", ".join([f"{val:.7f}" for val in results_dict['final_solution']])
    if error_norm <= tol:
        results_dict['conclusion'] = f"Convergió a x ≈ [{final_x_str}] en {k} iteraciones (Error={error_norm:.2e})."
    else:
        results_dict['conclusion'] = f"No convergió en {nmax} iteraciones. Última x ≈ [{final_x_str}] (Último error={error_norm:.2e})."
    
    return results_dict

# --- Funciones Envoltorio (Wrappers) para mantener la interfaz con views.py ---
def format_results_for_view(raw_output_dict):
    """
    Toma la salida de iterative_method_core y la formatea
    a la tupla que esperan las vistas.
    """
    formatted_iterations = []
    for iter_data in raw_output_dict.get('iterations', []):
        k, error_val, x_vec_list = iter_data
        formatted_iterations.append({
            'Iteración': k,
            'x_vector': x_vec_list, # Ya es una lista
            'Error': error_val if error_val is not None else np.nan
        })

    final_message_from_dict = raw_output_dict.get('conclusion', "Conclusión no disponible.")
    spectral_radius = raw_output_dict.get('spectral_radius', np.inf)
    # Usamos el mensaje de convergencia detallado que ya calculamos
    convergence_message_detail = raw_output_dict.get('convergence_message_detail', "Análisis de convergencia no disponible.")
    
    return formatted_iterations, final_message_from_dict, spectral_radius, convergence_message_detail

def method_jacobi(A, b, x0, tol, niter):
    raw_output = iterative_method_core(A, b, x0, tol, niter, method_type='jacobi')
    return format_results_for_view(raw_output)

def method_gauss_seidel(A, b, x0, tol, niter):
    raw_output = iterative_method_core(A, b, x0, tol, niter, method_type='gauss_seidel')
    return format_results_for_view(raw_output)

def method_sor(A, b, x0, tol, niter, omega):
    raw_output = iterative_method_core(A, b, x0, tol, niter, method_type='sor', w=omega)
    return format_results_for_view(raw_output)