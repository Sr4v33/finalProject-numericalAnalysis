# Importa las bibliotecas necesarias:
import numpy as np  # Para cálculos numéricos eficientes, especialmente con matrices y vectores.
import json         # Para convertir datos (como los de gráficos) a formato JSON para la web.
import sympy as sp  # Para matemáticas simbólicas: crear polinomios, formatearlos y generar LaTeX.

# --- SÍMBOLO GLOBAL y FUNCIÓN DE EVALUACIÓN ---

# Define 'x' como un símbolo matemático global usando Sympy.
# Esto nos permite construir expresiones polinómicas de forma simbólica (ej: 2*x + 1).
x_sym = sp.Symbol('x')

def evaluate_interpolation(poly_data, x_eval_val):
    """
    Evalúa un polinomio o spline (devuelto por las funciones calculate_)
    en un punto específico x_eval_val.

    Args:
        poly_data: Puede ser un objeto de polinomio Sympy o una lista de tuplas
                   (polinomio_sympy, x_inicio, x_fin) para splines.
        x_eval_val (float): El valor de X en el que se desea evaluar el polinomio/spline.

    Returns:
        float: El valor Y evaluado, o 'nan' (Not a Number) si ocurre un error.
    """
    try:
        # Verifica si poly_data es una lista, lo que indica que es un spline.
        if isinstance(poly_data, list): # Es un spline
            # Itera sobre cada tramo (polinomio, x_inicio, x_fin) del spline.
            for i, (poly, x_start, x_end) in enumerate(poly_data):
                # Verifica si es el último intervalo (para incluir el límite superior).
                is_last_interval = (i == len(poly_data) - 1)
                # Comprueba si x_eval_val cae dentro del intervalo actual.
                # Para el último intervalo, se incluye x_end (<=).
                if (x_start <= x_eval_val < x_end) or \
                   (is_last_interval and x_start <= x_eval_val <= x_end):
                    # Convierte el polinomio Sympy en una función numérica rápida usando lambdify.
                    func = sp.lambdify(x_sym, poly, 'numpy')
                    # Evalúa la función y devuelve el resultado como float.
                    return float(func(x_eval_val))

            # Si x_eval_val está fuera del rango principal, intenta una extrapolación simple.
            # (Usa el primer o último tramo). Esto es opcional pero puede evitar errores.
            if x_eval_val < poly_data[0][1]:
                 func = sp.lambdify(x_sym, poly_data[0][0], 'numpy')
                 return float(func(x_eval_val))
            elif x_eval_val > poly_data[-1][2]:
                 func = sp.lambdify(x_sym, poly_data[-1][0], 'numpy')
                 return float(func(x_eval_val))

            # Si no se encontró el intervalo, lanza un error (aunque la extrapolación lo evita a menudo).
            raise ValueError(f"x_eval={x_eval_val} está fuera del rango de interpolación.")
        
        # Si no es una lista, es un polinomio único (Vandermonde, Newton, Lagrange).
        else:
            # Convierte el polinomio Sympy en una función numérica.
            func = sp.lambdify(x_sym, poly_data, 'numpy')
            # Evalúa y devuelve el resultado.
            return float(func(x_eval_val))
            
    except Exception as e:
        # En caso de cualquier error durante la evaluación, imprime un mensaje y devuelve 'nan'.
        print(f"Error evaluando en {x_eval_val}: {e}")
        return float('nan')

def format_polynomial(coeffs, precision=6):
    """
    Formatea una lista de coeficientes de un polinomio en una cadena legible
    (ej: "1.00x^2 - 2.00x + 3.00"). No usa Sympy ni LaTeX.

    Args:
        coeffs (list): Lista de coeficientes, del término de mayor grado al menor.
        precision (int): Número de decimales para mostrar.

    Returns:
        str: El polinomio como una cadena de texto.
    """
    poly_str = []
    n = len(coeffs) - 1 # Grado del polinomio.
    
    # Itera sobre cada coeficiente y su índice.
    for i, a in enumerate(coeffs):
        power = n - i # Potencia de x para este término.

        # Ignora coeficientes muy cercanos a cero, a menos que sea el término constante.
        if abs(a) < 1e-9 and n != i : 
            continue

        # Formatea el coeficiente con la precisión dada.
        term = f"{a:.{precision}f}"

        # Añade 'x' y su potencia si es mayor que 0.
        if power > 0:
            term += f"x"
            if power > 1:
                term += f"^{power}"
        poly_str.append(term)
    
    # Si todos los coeficientes eran cero, devuelve "0.0".
    if not poly_str: return "0.0"

    # Construye la cadena final, manejando los signos '+' y '-'.
    result = poly_str[0]
    for term in poly_str[1:]:
        if term.startswith('-'):
            result += f" - {term[1:]}" # Si empieza con '-', añade ' - ' y quita el '-'.
        else:
            result += f" + {term}" # Si es positivo, añade ' + '.
    return result

# --- FUNCIONES calculate_... MODIFICADAS ---

def calculate_vandermonde(points):
    """
    Calcula la interpolación de Vandermonde.

    Devuelve:
        tuple: Contiene la matriz A, vector b, coeficientes, polinomio en string,
               datos para graficar, objeto Sympy del polinomio y string LaTeX.
    """
    # Extrae los valores X e Y de los puntos de entrada.
    x_values = np.array([p['x'] for p in points], dtype=float)
    y_values = np.array([p['y'] for p in points], dtype=float)
    n = len(x_values)
    
    # Crea la matriz de Vandermonde.
    matrix_a = np.vander(x_values, n, increasing=False)
    # Crea el vector de valores Y.
    vector_b = y_values.reshape(-1, 1)
    
    # Resuelve el sistema de ecuaciones lineales (A * coeffs = b) para encontrar los coeficientes.
    coeffs = np.linalg.solve(matrix_a, y_values)
    
    # Formatea el polinomio como una cadena de texto simple.
    poly_str_normal = format_polynomial(coeffs)

    # Genera datos para la gráfica: 100 puntos interpolados entre Xmin y Xmax.
    p_numpy = np.poly1d(coeffs) # Crea un objeto polinomio de Numpy para evaluar fácil.
    interpolated_x = np.linspace(np.min(x_values), np.max(x_values), 100).tolist()
    interpolated_y = p_numpy(interpolated_x).tolist()
    plot_data = json.dumps({
        'original_x': x_values.tolist(), 'original_y': y_values.tolist(),
        'interpolated_x': interpolated_x, 'interpolated_y': interpolated_y,
    })

    # --- AÑADIDO: SYMPY y LATEX ---
    # Construye el polinomio simbólico usando Sympy.
    poly_sympy = sum(c * x_sym**(len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    # Genera la representación LaTeX del polinomio (con 6 decimales).
    poly_str_latex = sp.latex(poly_sympy.evalf(6), mul_symbol=' ')
    # --- FIN AÑADIDO ---

    # Devuelve todos los resultados.
    return (
        matrix_a.tolist(), vector_b.tolist(), coeffs.tolist(),
        poly_str_normal, plot_data,
        poly_sympy, poly_str_latex # <-- Devuelve los nuevos valores.
    )

def calculate_newton(points):
    """
    Calcula la interpolación de Newton (Diferencias Divididas).

    Devuelve:
        tuple: Contiene cabeceras, tabla de diferencias, coeficientes de Newton,
               polinomio expandido, polinomio nativo, datos para graficar,
               objeto Sympy del polinomio y string LaTeX.
    """
    n = len(points)
    x_pts = np.array([p['x'] for p in points], dtype=float)
    y_pts = np.array([p['y'] for p in points], dtype=float)
    
    # Calcula la tabla de diferencias divididas.
    dd_table = np.zeros((n, n))
    dd_table[:, 0] = y_pts # La primera columna son los valores Y.
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i, j] = (dd_table[i + 1, j - 1] - dd_table[i, j - 1]) / (x_pts[i + j] - x_pts[i])
    # Los coeficientes de Newton son la primera fila de la tabla.
    newton_coeffs_divided_differences = dd_table[0, :]

    # Expande el polinomio de Newton a su forma estándar (a_n*x^n + ... + a_0).
    expanded_poly_coeffs_np = np.array([newton_coeffs_divided_differences[0]])
    product_term_poly = np.array([1.0])
    for i in range(1, n):
        current_x_factor = np.array([1.0, -x_pts[i-1]]) # (x - x_i)
        product_term_poly = np.polymul(product_term_poly, current_x_factor) # Multiplica por el siguiente factor.
        term_to_add = np.polymul(np.array([newton_coeffs_divided_differences[i]]), product_term_poly) # Coeff * (x-x0)...(x-xi)
        expanded_poly_coeffs_np = np.polyadd(expanded_poly_coeffs_np, term_to_add) # Suma al polinomio total.
    
    # Formatea el polinomio expandido y el nativo (forma de Newton).
    poly_str_expanded = format_polynomial(expanded_poly_coeffs_np.tolist())
    poly_str_newton_native = f"{newton_coeffs_divided_differences[0]:.5f}"
    current_product_factors_str = ""
    for i in range(1, n):
        coeff_val = newton_coeffs_divided_differences[i]
        prev_x = x_pts[i-1]
        current_product_factors_str += f"(x - {prev_x:g})" if prev_x >=0 else f"(x + {abs(prev_x):g})"
        poly_str_newton_native += f"{' + ' if coeff_val >= 0 else ' - '}{abs(coeff_val):.5f}{current_product_factors_str}"
    
    # Genera datos para la gráfica.
    plot_data = json.dumps({
        'original_x': x_pts.tolist(), 'original_y': y_pts.tolist(),
        'interpolated_x': np.linspace(min(x_pts), max(x_pts), 100).tolist(),
        'interpolated_y': np.poly1d(expanded_poly_coeffs_np)(np.linspace(min(x_pts), max(x_pts), 100)).tolist(),
    })
    # Prepara la tabla de diferencias para mostrarla.
    headers = ['n', 'x_i', 'y = f[xi]'] + [str(i) for i in range(1, n)]
    display_table = [[i, x_pts[i]] + [dd_table[i, j] if i <= n - 1 - j else 0.0 for j in range(n)] for i in range(n)]

    # --- AÑADIDO: SYMPY y LATEX ---
    # Construye el polinomio simbólico a partir de los coeficientes expandidos.
    poly_sympy = sum(c * x_sym**(len(expanded_poly_coeffs_np) - 1 - i) for i, c in enumerate(expanded_poly_coeffs_np))
    # Genera la representación LaTeX.
    poly_str_latex = sp.latex(poly_sympy.evalf(6), mul_symbol=' ')
    # --- FIN AÑADIDO ---

    return headers, display_table, newton_coeffs_divided_differences.tolist(), \
           poly_str_expanded, poly_str_newton_native, plot_data, \
           poly_sympy, poly_str_latex # <-- Devuelve los nuevos valores.

def calculate_lagrange(points):
    """
    Calcula la interpolación de Lagrange.

    Devuelve:
        tuple: Contiene polinomios base (placeholder), polinomio nativo (placeholder),
               polinomio expandido, datos para graficar,
               objeto Sympy del polinomio y string LaTeX.
    """
    n = len(points)
    x_pts = np.array([p['x'] for p in points], dtype=float)
    y_pts = np.array([p['y'] for p in points], dtype=float)

    # Inicializa el polinomio total como cero.
    total_expanded_poly_coeffs = np.array([0.0])
    
    # Itera para cada punto (i) para construir su polinomio base L_i(x).
    for i_loop_idx in range(n):
        # Calcula el numerador de L_i(x): el producto de (x - x_j) para j != i.
        li_numerator_poly = np.array([1.0])
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx: continue
            li_numerator_poly = np.polymul(li_numerator_poly, np.array([1.0, -x_pts[j_loop_idx]]))
        
        # Calcula el denominador de L_i(x): el producto de (x_i - x_j) para j != i.
        li_denominator_value = 1.0
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx: continue
            li_denominator_value *= (x_pts[i_loop_idx] - x_pts[j_loop_idx])
            
        # Calcula los coeficientes del polinomio base L_i(x).
        li_coeffs = li_numerator_poly / li_denominator_value
        # Multiplica L_i(x) por y_i.
        term_coeffs = np.polymul(np.array([y_pts[i_loop_idx]]), li_coeffs)
        # Suma este término al polinomio total.
        total_expanded_poly_coeffs = np.polyadd(total_expanded_poly_coeffs, term_coeffs)

    # Formatea el polinomio expandido.
    poly_str_expanded = format_polynomial(total_expanded_poly_coeffs.tolist())
    
    # Genera datos para la gráfica.
    plot_data = json.dumps({
        'original_x': x_pts.tolist(), 'original_y': y_pts.tolist(),
        'interpolated_x': np.linspace(min(x_pts), max(x_pts), 100).tolist(),
        'interpolated_y': np.poly1d(total_expanded_poly_coeffs)(np.linspace(min(x_pts), max(x_pts), 100)).tolist(),
    })
    
    # Placeholders para valores que no se calcularon explícitamente aquí.
    lagrange_basis_for_table = [] 
    poly_str_lagrange_native = "" 

    # --- AÑADIDO: SYMPY y LATEX ---
    # Construye el polinomio simbólico a partir de los coeficientes expandidos.
    poly_sympy = sum(c * x_sym**(len(total_expanded_poly_coeffs) - 1 - i) for i, c in enumerate(total_expanded_poly_coeffs))
    # Genera la representación LaTeX.
    poly_str_latex = sp.latex(poly_sympy.evalf(6), mul_symbol=' ')
    # --- FIN AÑADIDO ---

    return lagrange_basis_for_table, poly_str_lagrange_native, poly_str_expanded, plot_data, \
           poly_sympy, poly_str_latex # <-- Devuelve los nuevos valores.

def calculate_linear_spline(points_data):
    """
    Calcula la interpolación por Spline Lineal.

    Devuelve:
        tuple: Contiene tabla de coeficientes, lista de trazadores, datos para graficar,
               lista de objetos Sympy (uno por tramo) y string LaTeX.
    """
    # Ordena los puntos por X, esencial para splines.
    points_data.sort(key=lambda p: p['x'])
    x = np.array([p['x'] for p in points_data], dtype=float)
    y = np.array([p['y'] for p in points_data], dtype=float)
    n = len(x)

    coeffs_table_list = [] # Para mostrar coeficientes.
    tracers_list = []      # Para mostrar ecuaciones de tramos (string).
    plot_x_all = []        # Coordenadas X para la gráfica.
    plot_y_all = []        # Coordenadas Y para la gráfica.
    # --- AÑADIDO: SYMPY y LATEX ---
    spline_sympy_list = [] # Lista para almacenar objetos Sympy (polinomio, x_inicio, x_fin).
    tracers_latex_list = []# Lista para almacenar strings LaTeX de cada tramo.
    # --- FIN AÑADIDO ---

    # Itera sobre cada par de puntos para crear un segmento lineal.
    for i in range(n - 1):
        xi, yi = x[i], y[i]
        xi1, yi1 = x[i+1], y[i+1]
        
        # Calcula la pendiente (m_i) y la ordenada al origen (c_i).
        m_i = (yi1 - yi) / (xi1 - xi)
        c_i = yi - m_i * xi
        
        # Almacena coeficientes y la ecuación como string.
        coeffs_table_list.append({'i': i, 'coeff1': m_i, 'coeff2': c_i})
        tracer_str_normal = f"{m_i:.6f}x {'+' if c_i >= 0 else '-'} {abs(c_i):.6f}"
        tracers_list.append({'i': i, 'tracer_str': tracer_str_normal})

        # --- AÑADIDO: SYMPY y LATEX ---
        # Crea el polinomio Sympy para el tramo actual: S_i(x) = m_i * x + c_i.
        Si = m_i * x_sym + c_i
        # Lo añade a la lista junto con su intervalo.
        spline_sympy_list.append((Si, xi, xi1))
        # Crea y añade la versión LaTeX.
        tracers_latex_list.append(f"S_{{{i}}}(x) = {sp.latex(Si.evalf(6))}")
        # --- FIN AÑADIDO ---

        # Genera puntos para graficar este segmento.
        x_segment = np.linspace(xi, xi1, 50)
        y_segment = m_i * x_segment + c_i
        plot_x_all.extend(x_segment.tolist())
        plot_y_all.extend(y_segment.tolist())

    # Crea los datos para la gráfica en formato JSON.
    plot_data = json.dumps({
        'original_x': x.tolist(), 'original_y': y.tolist(),
        'interpolated_x': plot_x_all, 'interpolated_y': plot_y_all, 'type': 'spline'
    })
    
    # Une todos los strings LaTeX en uno solo para mostrar.
    poly_str_latex = "; \\; ".join(tracers_latex_list)

    return coeffs_table_list, tracers_list, plot_data, \
           spline_sympy_list, poly_str_latex # <-- Devuelve los nuevos valores.


def calculate_cubic_spline(points_data):
    """
    Calcula la interpolación por Spline Cúbico Natural.

    Devuelve:
        tuple: Contiene tabla de coeficientes, lista de trazadores (LaTeX),
               datos para graficar, lista de objetos Sympy (uno por tramo) y string LaTeX.
    """
    # Ordena los puntos por X.
    points_data.sort(key=lambda p: p['x'])
    x = np.array([p['x'] for p in points_data], dtype=float)
    y = np.array([p['y'] for p in points_data], dtype=float)
    n = len(x)
    
    # Calcula h_i = x_{i+1} - x_i.
    h = x[1:] - x[:-1]
    
    # Construye el sistema de ecuaciones tridiagonal (A * M = B)
    # para encontrar las segundas derivadas (M_i) en cada punto interior.
    A = np.zeros((n - 2, n - 2))
    B = np.zeros(n - 2)
    for i in range(n - 2):
        A[i, i] = 2 * (h[i] + h[i+1])
        B[i] = 6 * ((y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i])
        if i > 0: A[i, i-1] = h[i]
        if i < n - 3: A[i, i+1] = h[i+1]
        
    # Resuelve el sistema para obtener M_1 a M_{n-1}.
    M_internal = np.linalg.solve(A, B)
    # Añade M_0 = 0 y M_n = 0 (condición de spline natural).
    M = np.concatenate(([0], M_internal, [0]))
    
    # Calcula los coeficientes a, b, c, d para cada tramo cúbico
    # S_i(x) = a_i(x-x_i)^3 + b_i(x-x_i)^2 + c_i(x-x_i) + d_i.
    a = (M[1:] - M[:-1]) / (6 * h)
    b = M[:-1] / 2
    c = (y[1:] - y[:-1]) / h - h * (2 * M[:-1] + M[1:]) / 6
    d = y[:-1]

    coeffs_table_list = []
    tracers_list = []
    plot_x_all = []
    plot_y_all = []
    # --- AÑADIDO: SYMPY y LATEX ---
    spline_sympy_list = []
    tracers_latex_list = []
    # --- FIN AÑADIDO ---

    # Itera sobre cada tramo para construir y formatear los polinomios.
    for i in range(n - 1):
        # Construye el polinomio Sympy en la forma S_i(x) = a_i(x-x_i)^3 + ...
        Si = (a[i] * (x_sym - x[i])**3 + b[i] * (x_sym - x[i])**2 + c[i] * (x_sym - x[i]) + d[i])
        # Expande el polinomio a la forma A_i*x^3 + B_i*x^2 + C_i*x + D_i.
        Si_expanded = sp.expand(Si)
        
        # Extrae los coeficientes A, B, C, D del polinomio expandido.
        poly = sp.Poly(Si_expanded, x_sym)
        A_i = float(poly.coeff_monomial(x_sym**3))
        B_i = float(poly.coeff_monomial(x_sym**2))
        C_i = float(poly.coeff_monomial(x_sym**1))
        D_i = float(poly.coeff_monomial(x_sym**0))
        
        # Almacena los coeficientes y genera la versión LaTeX.
        coeffs_table_list.append({'i': i, 'coeff1': A_i, 'coeff2': B_i, 'coeff3': C_i, 'coeff4': D_i})
        tracer_str_latex = sp.latex(Si_expanded.evalf(6), mul_symbol=' ')
        tracers_list.append({'i': i, 'tracer_str': tracer_str_latex})

        # --- AÑADIDO: SYMPY y LATEX ---
        # Almacena el polinomio Sympy expandido y su intervalo.
        spline_sympy_list.append((Si_expanded, x[i], x[i+1]))
        # Almacena la cadena LaTeX del tramo.
        tracers_latex_list.append(f"S_{{{i}}}(x) = {tracer_str_latex}")
        # --- FIN AÑADIDO ---

        # Genera puntos para graficar este segmento cúbico.
        x_segment = np.linspace(x[i], x[i+1], 50)
        func_Si = sp.lambdify(x_sym, Si_expanded, 'numpy') # Convierte Sympy a función numérica.
        y_segment = func_Si(x_segment)
        plot_x_all.extend(x_segment.tolist())
        plot_y_all.extend(y_segment.tolist())

    # Crea los datos para la gráfica en formato JSON.
    plot_data = json.dumps({
        'original_x': x.tolist(), 'original_y': y.tolist(),
        'interpolated_x': plot_x_all, 'interpolated_y': plot_y_all, 'type': 'spline'
    })

    # Une todos los strings LaTeX en uno solo para mostrar.
    poly_str_latex = "; \\; ".join(tracers_latex_list)

    return coeffs_table_list, tracers_list, plot_data, \
           spline_sympy_list, poly_str_latex # <-- Devuelve los nuevos valores.