# core/chapterTwo/methods.py
import numpy as np

# --- Funciones Auxiliares ---

def calculate_spectral_radius(T):
    """
    Calcula el radio espectral (ρ) de una matriz de iteración T.
    El radio espectral es el máximo de los valores absolutos de los autovalores.
    Es un indicador clave para la convergencia de métodos iterativos (ρ(T) < 1).

    Args:
        T (np.ndarray): La matriz de iteración.

    Returns:
        float: El radio espectral, o np.inf si no se puede calcular.
    """
    try:
        # Si la matriz está vacía o es None, no se puede calcular.
        if T is None or T.size == 0:
            return np.inf
        # Calcula los autovalores de T.
        eigenvalues = np.linalg.eigvals(T)
        # Devuelve el máximo de los valores absolutos de los autovalores.
        return np.max(np.abs(eigenvalues))
    # Captura errores de álgebra lineal (ej: no converge el cálculo de autovalores).
    except np.linalg.LinAlgError:
        return np.inf
    # Captura cualquier otro error inesperado.
    except Exception:
        return np.inf

def get_D_L_U(A):
    """
    Descompone una matriz A en sus componentes: Diagonal (D),
    Estrictamente Triangular Inferior (L) y Estrictamente Triangular Superior (U).
    La descomposición se basa en la definición A = D - L - U.

    Args:
        A (np.ndarray): La matriz cuadrada de entrada.

    Returns:
        tuple: Una tupla conteniendo las matrices (D, L, U).
    """
    # D: Matriz con solo los elementos diagonales de A.
    D = np.diag(np.diag(A))
    # L: Matriz con los elementos debajo de la diagonal de A (con signo cambiado).
    L = -np.tril(A, -1) 
    # U: Matriz con los elementos encima de la diagonal de A (con signo cambiado).
    U = -np.triu(A, 1)  
    return D, L, U

# --- Núcleo del Método Iterativo ---

def iterative_method_core(A_in, b_in, x0_in, tol, nmax, method_type='jacobi', w=1.0):
    """
    Implementa el núcleo de los métodos iterativos (Jacobi, Gauss-Seidel, SOR)
    para resolver sistemas de ecuaciones lineales Ax = b.

    Args:
        A_in (list): Matriz A como lista de listas.
        b_in (list): Vector b como lista.
        x0_in (list): Vector inicial x0 como lista.
        tol (float): Tolerancia para el criterio de parada.
        nmax (int): Número máximo de iteraciones.
        method_type (str): Tipo de método ('jacobi', 'gauss_seidel', 'sor').
        w (float): Factor de relajación para SOR (0 < w < 2).

    Returns:
        dict: Un diccionario con todos los resultados, incluyendo matrices,
              radio espectral, historial de iteraciones y conclusiones.
    """
    # Convierte las entradas a arrays de NumPy para cálculos numéricos.
    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)
    x0 = np.array(x0_in, dtype=float)

    # Diccionario para almacenar todos los resultados generados.
    results_dict = {
        'D': None, 'L': None, 'U': None, 'T': None, 'C': None,
        'spectral_radius': np.inf, # Radio espectral, default a infinito.
        'iterations': [], # Lista para guardar [iteración, error, vector_x].
        'conclusion': "El método no se ejecutó completamente.", # Mensaje final.
        'final_solution': None, # Solución encontrada.
        'convergence_message_detail': "No determinado." # Mensaje sobre ρ(T).
    }

    # --- Validaciones Iniciales ---
    # Verifica si hay ceros en la diagonal (crítico para D invertible en Jacobi/GS).
    if np.any(np.diag(A) == 0):
        if method_type == 'jacobi' or method_type == 'gauss_seidel':
            results_dict['conclusion'] = "Fracasó: La matriz tiene ceros en su diagonal principal."
            results_dict['convergence_message_detail'] = "No converge: Elemento diagonal cero."
            return results_dict
    
    # Verifica si la matriz es singular (determinante cero o muy cercano).
    try:
        if abs(np.linalg.det(A)) < 1e-12: 
            results_dict['conclusion'] = "Fracasó: det(A) es (o muy cercano a) 0. Matriz singular."
            results_dict['convergence_message_detail'] = "No converge: Matriz singular."
            return results_dict
    except np.linalg.LinAlgError:
        results_dict['conclusion'] = "Fracasó: No se pudo calcular det(A) (posiblemente no cuadrada)."
        results_dict['convergence_message_detail'] = "Error de matriz."
        return results_dict

    # Verifica si la tolerancia y las iteraciones máximas son válidas.
    if tol <= 0 or nmax <= 0: # Corregido a <= 0
        results_dict['conclusion'] = "Fracasó: Tolerancia o Nmax inválidos (deben ser > 0)."
        return results_dict
    
    # Verifica si omega (w) es válido para SOR.
    if method_type == 'sor' and not (0 < w < 2):
        results_dict['conclusion'] = f"Fracasó: Omega (ω) para SOR debe estar en (0, 2), se recibió ω={w}."
        results_dict['convergence_message_detail'] = "Omega fuera de rango."
        return results_dict

    # --- Preparación del Método ---
    # Descompone A en D, L, U y las guarda.
    D, L, U = get_D_L_U(A)
    results_dict['D'], results_dict['L'], results_dict['U'] = D.tolist(), L.tolist(), U.tolist()

    T = None    # Matriz de iteración (T).
    C_vec = None # Vector de iteración (C).

    # Calcula T y C según el método seleccionado.
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
            D_minus_wL = D - w * L
            D_minus_wL_inv = np.linalg.inv(D_minus_wL)
            T = np.dot(D_minus_wL_inv, ((1 - w) * D + w * U))
            C_vec = w * np.dot(D_minus_wL_inv, b)
        else:
            results_dict['conclusion'] = f"Fracasó: método no reconocido '{method_type}'."
            return results_dict
    # Captura errores al invertir matrices (singularidad).
    except np.linalg.LinAlgError as e:
        results_dict['conclusion'] = f"Fracasó: Matriz singular durante cálculo de T y C ({e})."
        results_dict['convergence_message_detail'] = f"Error de álgebra lineal: {e}"
        return results_dict

    # Guarda T y C (convertidos a lista) en los resultados.
    results_dict['T'] = T.tolist() if T is not None else None
    results_dict['C_vector_calc'] = C_vec.tolist() if C_vec is not None else None 
    
    # Calcula y guarda el radio espectral y genera un mensaje sobre la convergencia teórica.
    current_spectral_radius = calculate_spectral_radius(T)
    results_dict['spectral_radius'] = current_spectral_radius

    if current_spectral_radius == np.inf:
        results_dict['convergence_message_detail'] = "No se pudo calcular el radio espectral."
    elif current_spectral_radius >= 1:
        results_dict['convergence_message_detail'] = f"ADVERTENCIA: ρ(T) = {current_spectral_radius:.7f} ≥ 1. El método puede no converger."
    else:
        results_dict['convergence_message_detail'] = f"Converge teóricamente (ρ(T) = {current_spectral_radius:.7f} < 1)."

    # --- Proceso Iterativo ---
    x_prev = x0.copy()
    # Guarda la iteración 0 (punto inicial).
    results_dict['iterations'].append([0, np.nan, x_prev.copy().tolist()])

    error_norm = tol + 1 # Inicializa el error mayor que la tolerancia para entrar al bucle.
    k = 0 # Contador de iteraciones.

    # Bucle principal: itera mientras el error sea mayor que la tolerancia y no se alcance nmax.
    while error_norm > tol and k < nmax:
        # Si T o C no se pudieron calcular, detiene el bucle.
        if T is None or C_vec is None: 
            results_dict['conclusion'] = "Fracasó: Matrices T o C no calculadas."
            break
        
        # Calcula la nueva aproximación: x_k+1 = T * x_k + C.
        x_current = np.dot(T, x_prev) + C_vec
        
        # Calcula el error como la norma infinita de la diferencia entre iteraciones.
        error_norm = np.max(np.abs(x_current - x_prev))
        
        k += 1 # Incrementa el contador.
        # Guarda la iteración actual.
        results_dict['iterations'].append([k, error_norm, x_current.tolist()])
        # Actualiza x_prev para la siguiente iteración.
        x_prev = x_current.copy()

    # Guarda la solución final.
    results_dict['final_solution'] = x_prev.tolist() 
    
    # Genera el mensaje de conclusión final.
    final_x_str = ", ".join([f"{val:.7f}" for val in results_dict['final_solution']])
    if error_norm <= tol:
        results_dict['conclusion'] = f"Convergió a x ≈ [{final_x_str}] en {k} iteraciones (Error={error_norm:.2e})."
    else:
        results_dict['conclusion'] = f"No convergió en {nmax} iteraciones. Última x ≈ [{final_x_str}] (Último error={error_norm:.2e})."
    
    # Devuelve el diccionario completo de resultados.
    return results_dict

# --- Funciones Envoltorio (Wrappers) ---

def format_results_for_view(raw_output_dict):
    """
    Toma el diccionario de resultados de iterative_method_core y lo formatea
    a la tupla que esperan las vistas (views.py) para mantener la compatibilidad.

    Args:
        raw_output_dict (dict): El diccionario devuelto por iterative_method_core.

    Returns:
        tuple: (iterations_list, conclusion_message, spectral_radius, convergence_message)
    """
    formatted_iterations = []
    # Itera sobre las iteraciones guardadas y les da un formato de diccionario.
    for iter_data in raw_output_dict.get('iterations', []):
        k, error_val, x_vec_list = iter_data
        formatted_iterations.append({
            'Iteración': k,
            'x_vector': x_vec_list,
            'Error': error_val if not np.isnan(error_val) else 'N/A' # Muestra N/A para la iteración 0.
        })

    # Extrae los mensajes y el radio espectral.
    final_message_from_dict = raw_output_dict.get('conclusion', "Conclusión no disponible.")
    spectral_radius = raw_output_dict.get('spectral_radius', np.inf)
    convergence_message_detail = raw_output_dict.get('convergence_message_detail', "Análisis no disponible.")
    
    # Devuelve la tupla formateada.
    return formatted_iterations, final_message_from_dict, spectral_radius, convergence_message_detail

def method_jacobi(A, b, x0, tol, niter):
    """Función envoltorio para el método de Jacobi."""
    # Llama al núcleo con 'jacobi'.
    raw_output = iterative_method_core(A, b, x0, tol, niter, method_type='jacobi')
    # Formatea la salida.
    return format_results_for_view(raw_output)

def method_gauss_seidel(A, b, x0, tol, niter):
    """Función envoltorio para el método de Gauss-Seidel."""
    # Llama al núcleo con 'gauss_seidel'.
    raw_output = iterative_method_core(A, b, x0, tol, niter, method_type='gauss_seidel')
    # Formatea la salida.
    return format_results_for_view(raw_output)

def method_sor(A, b, x0, tol, niter, omega):
    """Función envoltorio para el método SOR."""
    # Llama al núcleo con 'sor' y el valor de omega.
    raw_output = iterative_method_core(A, b, x0, tol, niter, method_type='sor', w=omega)
    # Formatea la salida.
    return format_results_for_view(raw_output)