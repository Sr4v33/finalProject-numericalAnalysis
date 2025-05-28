# chapterThree/views.py (o un archivo dedicado como chapterThree/methods.py)

import numpy as np
import json
from scipy.interpolate import CubicSpline

def format_polynomial(coeffs, precision=6):
    """Formatea los coeficientes en una cadena de polinomio legible."""
    poly_str = []
    n = len(coeffs) - 1
    for i, a in enumerate(coeffs):
        power = n - i
        # Ignorar términos con coeficiente cercano a cero (opcional)
        # if abs(a) < 1e-9:
        #    continue

        term = f"{a:.{precision}f}"

        if power > 0:
            term += f"x"
            if power > 1:
                term += f"^{power}"

        poly_str.append(term)

    # Unir términos con signos +/-
    result = poly_str[0]
    for term in poly_str[1:]:
        if term.startswith('-'):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    return result

def calculate_vandermonde(points):
    """
    Calcula la interpolación de Vandermonde.
    Retorna: matrix_a, vector_b, coeffs, poly_str, plot_data
    Lanza: np.linalg.LinAlgError si la matriz es singular.
    Lanza: ValueError si los puntos no son válidos.
    """
    if not points or len(points) < 2:
        raise ValueError("Se necesitan al menos 2 puntos.")

    x_values = np.array([p['x'] for p in points], dtype=float)
    y_values = np.array([p['y'] for p in points], dtype=float)

    if len(x_values) != len(np.unique(x_values)):
         raise ValueError("Los valores de X deben ser únicos.")

    n = len(x_values)

    # Construir la matriz de Vandermonde (A)
    # np.vander(x, n, increasing=False) crea [x^(n-1), x^(n-2), ..., x^0]
    matrix_a = np.vander(x_values, n, increasing=False)

    # Construir el vector B (y_values)
    vector_b = y_values.reshape(-1, 1) # Vector columna

    # Resolver el sistema Ax = B para obtener los coeficientes (a)
    # np.linalg.solve puede lanzar LinAlgError si A es singular
    coeffs = np.linalg.solve(matrix_a, y_values)

    # Generar el polinomio como string
    poly_str = format_polynomial(coeffs)

    # Generar datos para graficar
    original_x = x_values.tolist()
    original_y = y_values.tolist()
    
    # Crear puntos interpolados para una curva suave
    x_min, x_max = np.min(original_x), np.max(original_x)
    interpolated_x = np.linspace(x_min, x_max, 100).tolist()
    
    # Crear el polinomio de numpy para evaluar fácilmente
    p_numpy = np.poly1d(coeffs)
    interpolated_y = p_numpy(interpolated_x).tolist()

    plot_data = json.dumps({
        'original_x': original_x,
        'original_y': original_y,
        'interpolated_x': interpolated_x,
        'interpolated_y': interpolated_y,
    })

    # Devolvemos todo lo necesario para el template
    return (
        matrix_a.tolist(),  # Convertir a lista para el template
        vector_b.tolist(),  # Convertir a lista para el template
        coeffs.tolist(),    # Convertir a lista para el template
        poly_str,
        plot_data
    )


def calculate_newton(points):
    n = len(points)
    x_pts = np.array([p['x'] for p in points], dtype=float)
    y_pts = np.array([p['y'] for p in points], dtype=float)

    if len(x_pts) != len(np.unique(x_pts)):
         raise ValueError("Los valores de X deben ser únicos para Newton.")

    dd_table = np.zeros((n, n))
    dd_table[:, 0] = y_pts
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i, j] = (dd_table[i + 1, j - 1] - dd_table[i, j - 1]) / (x_pts[i + j] - x_pts[i])

    newton_coeffs_divided_differences = dd_table[0, :] # Estos son f[x0], f[x0,x1], ...

    # --- LÓGICA PARA EXPANDIR EL POLINOMIO (como en la respuesta anterior) ---
    expanded_poly_coeffs_np = np.array([newton_coeffs_divided_differences[0]]) 
    product_term_poly = np.array([1.0]) 

    for i in range(1, n):
        current_x_factor = np.array([1.0, -x_pts[i-1]])
        product_term_poly = np.polymul(product_term_poly, current_x_factor)
        term_to_add = np.polymul(np.array([newton_coeffs_divided_differences[i]]), product_term_poly)
        expanded_poly_coeffs_np = np.polyadd(expanded_poly_coeffs_np, term_to_add)
    
    # expanded_poly_coeffs_np ahora tiene los coeficientes en forma a_n x^n + ... + a_0
    # ¡Llamamos a TU función format_polynomial!
    poly_str_expanded = format_polynomial(expanded_poly_coeffs_np.tolist())
    # --- FIN DE LA LÓGICA DE EXPANSIÓN Y FORMATEO ---


     # --- NUEVA LÓGICA PARA LA FORMA NATIVA DE NEWTON ---
    precision_newton_form = 5 # Precisión para los coeficientes en la forma de Newton, como en la imagen
    poly_str_newton_native = f"{newton_coeffs_divided_differences[0]:.{precision_newton_form}f}"
    current_product_factors_str = ""
    for i in range(1, n):
        coeff_val = newton_coeffs_divided_differences[i]
        prev_x = x_pts[i-1]

        # Formatear el factor (x - x_k)
        # Usamos :g para un formato limpio de los x_k (ej: 1 en lugar de 1.0)
        if prev_x == 0:
            current_product_factors_str += "(x)" 
        elif prev_x < 0:
            current_product_factors_str += f"(x + {abs(prev_x):g})"
        else:
            current_product_factors_str += f"(x - {prev_x:g})"

        sign = " + " if coeff_val >= 0 else " - "
        # El formato de la imagen parece tener el signo, luego el coeficiente, luego los términos (x-xk)
        poly_str_newton_native += f"{sign}{abs(coeff_val):.{precision_newton_form}f}{current_product_factors_str}"
    # --- FIN DE LA NUEVA LÓGICA PARA FORMA NATIVA ---


    # Generar datos para graficar (usando Horner para Newton)
    def eval_newton(x_eval, x_pts_eval, coeffs_eval): # coeffs_eval son las diferencias divididas
        n_eval = len(coeffs_eval)
        p = coeffs_eval[n_eval - 1]
        for k in range(1, n_eval):
            p = coeffs_eval[n_eval - 1 - k] + (x_eval - x_pts_eval[n_eval - 1 - k]) * p
        return p

    x_min, x_max = np.min(x_pts), np.max(x_pts)
    interpolated_x = np.linspace(x_min, x_max, 100).tolist()
    # Para eval_newton, usamos los coeficientes de Newton (diferencias divididas), no los expandidos
    interpolated_y = [eval_newton(val, x_pts, newton_coeffs_divided_differences) for val in interpolated_x]


    plot_data = json.dumps({
        'original_x': x_pts.tolist(),
        'original_y': y_pts.tolist(),
        'interpolated_x': interpolated_x,
        'interpolated_y': interpolated_y,
    })

    # Preparar la tabla para mostrarla (similar a la imagen)
    headers = ['n', 'x_i', 'y = f[xi]'] + [str(i) for i in range(1, n)]
    display_table = []
    for i in range(n):
        row = [i, x_pts[i]]
        # Añadir las columnas de la tabla calculada
        for j in range(n):
             if i <= n - 1 - j:
                 row.append(dd_table[i, j])
             else:
                 # Mostrar None o 0.0 según prefieras (la imagen muestra 0.0)
                 row.append(0.0) 
        display_table.append(row)

    return headers, display_table, newton_coeffs_divided_differences.tolist(), poly_str_expanded, poly_str_newton_native, plot_data



def calculate_lagrange(points):

    # --- INICIO DE LA FUNCIÓN AUXILIAR (puede estar aquí o fuera) ---
    def _format_factor_str(val1_display, val2_numeric):
        op = "-"
        val2_abs = abs(val2_numeric)
        if val2_numeric < 0:
            op = "+"
        # Mantenemos (x - 0) como en la imagen de ejemplo para consistencia
        # if val2_numeric == 0 and val1_display == "x":
        #     return "(x)" 
        return f"({val1_display} {op} {val2_abs:g})"
    # --- FIN DE LA FUNCIÓN AUXILIAR ---

    n = len(points)
    x_pts = np.array([p['x'] for p in points], dtype=float)
    y_pts = np.array([p['y'] for p in points], dtype=float)

    if len(x_pts) != len(np.unique(x_pts)):
        raise ValueError("Los valores de X deben ser únicos para Lagrange.")

    # 1. Generar strings para la tabla de Polinomios Base L_i(x)
    lagrange_basis_for_table = []
    for i_loop_idx in range(n): # Renombrado i para evitar conflicto con el nombre del índice 'i' en el dict
        # Numerador (x-x_j)...
        numerator_terms_list = []
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx:
                continue
            numerator_terms_list.append(_format_factor_str("x", x_pts[j_loop_idx]))
        numerator_str = "".join(numerator_terms_list)
        
        # Denominador ((x_i-x_j)...)
        denominator_terms_list = []
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx:
                continue
            denominator_terms_list.append(_format_factor_str(f"{x_pts[i_loop_idx]:g}", x_pts[j_loop_idx]))
        denominator_str_factors = "".join(denominator_terms_list)
            
        li_str_for_table = f"{numerator_str}/({denominator_str_factors})"
        lagrange_basis_for_table.append({'i': i_loop_idx, 'Li_x_str': li_str_for_table})

    # 2. Generar string para el Polinomio de Lagrange (Forma Nativa)
    native_poly_terms = []
    for i_loop_idx in range(n):
        numerator_terms_list_for_poly = []
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx:
                continue
            numerator_terms_list_for_poly.append(_format_factor_str("x", x_pts[j_loop_idx]))
        current_L_num_str = "".join(numerator_terms_list_for_poly)

        denominator_terms_list_for_poly = []
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx:
                continue
            denominator_terms_list_for_poly.append(_format_factor_str(f"{x_pts[i_loop_idx]:g}", x_pts[j_loop_idx]))
        current_L_den_str = "".join(denominator_terms_list_for_poly)
        
        # Formato del término: (yi * Numerador_Li_str) / (Denominador_Li_str)
        # Usamos :g para y_i también para limpieza si es entero
        term_str = f"({y_pts[i_loop_idx]:g}*{current_L_num_str})/({current_L_den_str})"
        native_poly_terms.append(term_str)
    
    poly_str_lagrange_native = native_poly_terms[0]
    for k_loop_idx in range(1, len(native_poly_terms)): # Renombrado k
        poly_str_lagrange_native += f" + {native_poly_terms[k_loop_idx]}"


    # 3. Generar Polinomio Expandido (sin cambios en esta lógica)
    total_expanded_poly_coeffs = np.array([0.0])
    for i_loop_idx in range(n):
        li_numerator_poly = np.array([1.0])
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx:
                continue
            li_numerator_poly = np.polymul(li_numerator_poly, np.array([1.0, -x_pts[j_loop_idx]]))
        
        li_denominator_value = 1.0
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx:
                continue
            li_denominator_value *= (x_pts[i_loop_idx] - x_pts[j_loop_idx])
        
        if abs(li_denominator_value) < 1e-12:
            raise ValueError(f"Denominador de L_{i_loop_idx}(x) es cero.")
            
        li_coeffs = li_numerator_poly / li_denominator_value
        term_coeffs = np.polymul(np.array([y_pts[i_loop_idx]]), li_coeffs)
        total_expanded_poly_coeffs = np.polyadd(total_expanded_poly_coeffs, term_coeffs)
        
    poly_str_expanded = format_polynomial(total_expanded_poly_coeffs.tolist(), precision=6)

    # 4. Generar datos para graficar (sin cambios en esta lógica)
    p_expanded_func = np.poly1d(total_expanded_poly_coeffs)
    x_min, x_max = np.min(x_pts), np.max(x_pts)
    interpolated_x = np.linspace(x_min, x_max, 100).tolist()
    interpolated_y = p_expanded_func(interpolated_x).tolist()
    plot_data = json.dumps({
        'original_x': x_pts.tolist(), 'original_y': y_pts.tolist(),
        'interpolated_x': interpolated_x, 'interpolated_y': interpolated_y,
    })

    return lagrange_basis_for_table, poly_str_lagrange_native, poly_str_expanded, plot_data


def calculate_linear_spline(points):
    n = len(points)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 puntos para el spline lineal.")

    x_pts = np.array([p['x'] for p in points], dtype=float)
    y_pts = np.array([p['y'] for p in points], dtype=float)

    if len(x_pts) != len(np.unique(x_pts)):
        sorted_indices = np.argsort(x_pts)
        x_pts = x_pts[sorted_indices]
        y_pts = y_pts[sorted_indices]
        if len(x_pts) != len(np.unique(x_pts)):
            raise ValueError("Los valores de X deben ser únicos y preferiblemente ordenados.")

    coeffs_table = []
    tracers_list = []
    
    # Plotting data
    all_interpolated_x = []
    all_interpolated_y = []

    for i in range(n - 1):
        xi = x_pts[i]
        yi = y_pts[i]
        xi_plus_1 = x_pts[i+1]
        yi_plus_1 = y_pts[i+1]

        if abs(xi_plus_1 - xi) < 1e-9: # Evitar división por cero
            raise ValueError(f"Puntos X duplicados o muy cercanos en el índice {i} y {i+1}, no se puede calcular la pendiente.")

        # Coeff1: pendiente (m_i)
        m_i = (yi_plus_1 - yi) / (xi_plus_1 - xi)
        
        # Coeff2: intersección y (c_i), S_i(x) = m_i * x + c_i  =>  c_i = yi - m_i * xi
        c_i = yi - m_i * xi 
        
        coeffs_table.append({'i': i, 'coeff1': m_i, 'coeff2': c_i})
        
        # Tracer string: S_i(x) = m_i * x + c_i
        # Formato con signo para c_i
        sign_c = "+" if c_i >= 0 else "-"
        tracer_str = f"{m_i:.{9}f}x {sign_c} {abs(c_i):.{9}f}" # Usamos 9 decimales
        tracers_list.append({'i': i, 'tracer_str': tracer_str})


    linear_segments_for_plot = []
    for i in range(n - 1):
        xi, yi = float(x_pts[i]), float(y_pts[i]) # Asegurar que sean floats para JSON
        xi_plus_1, yi_plus_1 = float(x_pts[i+1]), float(y_pts[i+1])
        linear_segments_for_plot.append([[xi, yi], [xi_plus_1, yi_plus_1]])

    plot_data_json = json.dumps({
        'original_x': x_pts.tolist(), 
        'original_y': y_pts.tolist(),
        'linear_segments': linear_segments_for_plot # Nueva estructura
    })

    return coeffs_table, tracers_list, plot_data_json



def calculate_cubic_spline(points):
    N = len(points)
    if N < 2:
        raise ValueError("Se necesitan al menos 2 puntos para el spline cúbico.")

    x_pts = np.array([p['x'] for p in points], dtype=float)
    y_pts = np.array([p['y'] for p in points], dtype=float)

    # Asegurar que los puntos X estén ordenados (requerido por CubicSpline)
    sorted_indices = np.argsort(x_pts)
    x_pts = x_pts[sorted_indices]
    y_pts = y_pts[sorted_indices]

    if len(x_pts) != len(np.unique(x_pts)):
        raise ValueError("Los valores de X deben ser únicos.")

    # Caso especial para N=2: es una línea.
    # Scipy CubicSpline con bc_type='natural' maneja N=2 y N=3 adecuadamente.
    # Para N=2, produce una línea. Para N=3, también es natural.
    # Si quieres que la salida de N=2 sea idéntica a tu calculate_linear_spline
    # podrías mantener el if, pero Scipy lo manejará. Por simplicidad, dejaremos que Scipy lo haga.

    # Crear el spline cúbico natural usando scipy
    # bc_type='natural' establece las segundas derivadas en los extremos a cero.
    try:
        cs = CubicSpline(x_pts, y_pts, bc_type='natural')
    except ValueError as e:
        # Scipy puede lanzar errores si los datos no son adecuados (ej: x no monótono)
        raise ValueError(f"Error al crear el spline cúbico con Scipy: {e}")

    coeffs_table_display = [] # Para la tabla Coeff1-4
    tracers_list = []         # Para las cadenas de los trazadores

    all_interpolated_x = []
    all_interpolated_y = []

    # Iterar sobre cada segmento del spline (N-1 segmentos)
    # El atributo cs.c es una matriz de 4x(N-1)
    # Para el segmento i (entre x_pts[i] y x_pts[i+1]):
    # S_i(x) = c[0,i]*(x-x_i)^3 + c[1,i]*(x-x_i)^2 + c[2,i]*(x-x_i) + c[3,i]
    for i in range(N - 1):
        # Coeficientes del spline para el i-ésimo intervalo, relativos a (x - x_pts[i])
        # D_spl_local = cs.c[0, i]  # Coeficiente de (x-x_i)^3
        # C_spl_local = cs.c[1, i]  # Coeficiente de (x-x_i)^2
        # B_spl_local = cs.c[2, i]  # Coeficiente de (x-x_i)
        # A_spl_local = cs.c[3, i]  # Coeficiente constante (y_i)
        
        # Para evitar confusión con la nomenclatura A, B, C, D de la forma global,
        # usemos d_loc, c_loc, b_loc, a_loc para los coeficientes locales.
        d_loc = cs.c[0, i]
        c_loc = cs.c[1, i]
        b_loc = cs.c[2, i]
        a_loc = cs.c[3, i] # que es y_pts[i]

        # Convertir a coeficientes globales Ax^3 + Bx^2 + Cx + D para este segmento
        # S_i(x) = a_loc + b_loc(x-x_i) + c_loc(x-x_i)^2 + d_loc(x-x_i)^3
        
        # Coeff1 (para x^3)
        glob_A = d_loc
        # Coeff2 (para x^2)
        glob_B = c_loc - 3 * d_loc * x_pts[i]
        # Coeff3 (para x)
        glob_C = b_loc - 2 * c_loc * x_pts[i] + 3 * d_loc * x_pts[i]**2
        # Coeff4 (constante)
        glob_D = a_loc - b_loc * x_pts[i] + c_loc * x_pts[i]**2 - d_loc * x_pts[i]**3
        
        coeffs_table_display.append({
            'i': i,
            'coeff1': glob_A, # x^3
            'coeff2': glob_B, # x^2
            'coeff3': glob_C, # x
            'coeff4': glob_D  # const
        })
        
        # Usar format_polynomial para la cadena del tracer
        # Los coeficientes deben estar en orden [glob_A, glob_B, glob_C, glob_D]
        tracer_str = format_polynomial([glob_A, glob_B, glob_C, glob_D], precision=6)
        tracers_list.append({'i': i, 'tracer_str': tracer_str})

        # Para graficar el segmento actual
        x_segment = np.linspace(x_pts[i], x_pts[i+1], 20) # Más puntos para cúbico
        y_segment = cs(x_segment) # Scipy spline puede ser llamado como una función
        
        if i > 0: # Evitar duplicar puntos de conexión al concatenar segmentos
            all_interpolated_x.extend(x_segment[1:])
            all_interpolated_y.extend(y_segment[1:])
        else:
            all_interpolated_x.extend(x_segment)
            all_interpolated_y.extend(y_segment)

    plot_data = json.dumps({
        'original_x': x_pts.tolist(),
        'original_y': y_pts.tolist(),
        'interpolated_x': all_interpolated_x,
        'interpolated_y': all_interpolated_y,
    })

    return coeffs_table_display, tracers_list, plot_data