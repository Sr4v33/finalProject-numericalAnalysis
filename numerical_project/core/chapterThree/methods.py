import numpy as np
import json
import sympy as sp

# --- SÍMBOLO GLOBAL y FUNCIÓN DE EVALUACIÓN (Ya los tienes, asegúrate de que estén) ---
x_sym = sp.Symbol('x')

def evaluate_interpolation(poly_data, x_eval_val):
    """
    Evalúa un polinomio o spline (devuelto por las funciones calculate_)
    en un punto x_eval_val.
    """
    try:
        if isinstance(poly_data, list): # Es un spline
            for poly, x_start, x_end in poly_data:
                is_last_interval = (poly_data.index((poly, x_start, x_end)) == len(poly_data) - 1)
                if (x_start <= x_eval_val < x_end) or \
                   (is_last_interval and x_start <= x_eval_val <= x_end):
                    func = sp.lambdify(x_sym, poly, 'numpy')
                    return float(func(x_eval_val))
            # Extrapolación simple (opcional, pero ayuda a evitar errores)
            if x_eval_val < poly_data[0][1]:
                 func = sp.lambdify(x_sym, poly_data[0][0], 'numpy')
                 return float(func(x_eval_val))
            elif x_eval_val > poly_data[-1][2]:
                 func = sp.lambdify(x_sym, poly_data[-1][0], 'numpy')
                 return float(func(x_eval_val))
            raise ValueError(f"x_eval={x_eval_val} está fuera del rango de interpolación.")
        else: # Es un polinomio único
            func = sp.lambdify(x_sym, poly_data, 'numpy')
            return float(func(x_eval_val))
    except Exception as e:
        print(f"Error evaluando en {x_eval_val}: {e}")
        return float('nan')

def format_polynomial(coeffs, precision=6):
    """Formatea los coeficientes en una cadena de polinomio legible (como string normal)."""
    poly_str = []
    n = len(coeffs) - 1
    for i, a in enumerate(coeffs):
        power = n - i
        if abs(a) < 1e-9 and n != i : # Ignora coeficientes cero excepto el término constante
           continue

        term = f"{a:.{precision}f}"

        if power > 0:
            term += f"x"
            if power > 1:
                term += f"^{power}"
        poly_str.append(term)
    
    if not poly_str: return "0.0" # Si todos son cero

    result = poly_str[0]
    for term in poly_str[1:]:
        if term.startswith('-'):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    return result

# --- FUNCIONES calculate_... MODIFICADAS ---

def calculate_vandermonde(points):
    """ Calcula Vandermonde y devuelve datos + objeto Sympy + LaTeX. """
    x_values = np.array([p['x'] for p in points], dtype=float)
    y_values = np.array([p['y'] for p in points], dtype=float)
    n = len(x_values)
    # ... (Validaciones) ...
    matrix_a = np.vander(x_values, n, increasing=False)
    vector_b = y_values.reshape(-1, 1)
    coeffs = np.linalg.solve(matrix_a, y_values)
    poly_str_normal = format_polynomial(coeffs)
    # ... (Cálculo de plot_data) ...
    p_numpy = np.poly1d(coeffs)
    interpolated_x = np.linspace(np.min(x_values), np.max(x_values), 100).tolist()
    interpolated_y = p_numpy(interpolated_x).tolist()
    plot_data = json.dumps({
        'original_x': x_values.tolist(), 'original_y': y_values.tolist(),
        'interpolated_x': interpolated_x, 'interpolated_y': interpolated_y,
    })

    # --- AÑADIDO: SYMPY y LATEX ---
    poly_sympy = sum(c * x_sym**(len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    poly_str_latex = sp.latex(poly_sympy.evalf(6), mul_symbol=' ')
    # --- FIN AÑADIDO ---

    return (
        matrix_a.tolist(), vector_b.tolist(), coeffs.tolist(),
        poly_str_normal, plot_data,
        poly_sympy, poly_str_latex # <-- NUEVOS VALORES
    )



def calculate_newton(points):
    """ Calcula Newton y devuelve datos + objeto Sympy + LaTeX. """
    n = len(points)
    x_pts = np.array([p['x'] for p in points], dtype=float)
    y_pts = np.array([p['y'] for p in points], dtype=float)
    # ... (Validaciones) ...
    # ... (Cálculo de tabla dd_table y newton_coeffs_divided_differences) ...
    dd_table = np.zeros((n, n))
    dd_table[:, 0] = y_pts
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i, j] = (dd_table[i + 1, j - 1] - dd_table[i, j - 1]) / (x_pts[i + j] - x_pts[i])
    newton_coeffs_divided_differences = dd_table[0, :]

    # ... (Cálculo de expanded_poly_coeffs_np, poly_str_expanded, poly_str_newton_native, plot_data) ...
    expanded_poly_coeffs_np = np.array([newton_coeffs_divided_differences[0]])
    product_term_poly = np.array([1.0])
    for i in range(1, n):
        current_x_factor = np.array([1.0, -x_pts[i-1]])
        product_term_poly = np.polymul(product_term_poly, current_x_factor)
        term_to_add = np.polymul(np.array([newton_coeffs_divided_differences[i]]), product_term_poly)
        expanded_poly_coeffs_np = np.polyadd(expanded_poly_coeffs_np, term_to_add)
    poly_str_expanded = format_polynomial(expanded_poly_coeffs_np.tolist())
    poly_str_newton_native = f"{newton_coeffs_divided_differences[0]:.5f}"
    current_product_factors_str = ""
    for i in range(1, n):
        coeff_val = newton_coeffs_divided_differences[i]
        prev_x = x_pts[i-1]
        current_product_factors_str += f"(x - {prev_x:g})" if prev_x >=0 else f"(x + {abs(prev_x):g})"
        poly_str_newton_native += f"{' + ' if coeff_val >= 0 else ' - '}{abs(coeff_val):.5f}{current_product_factors_str}"
    
    plot_data = json.dumps({ # Simplified plot data calc
        'original_x': x_pts.tolist(), 'original_y': y_pts.tolist(),
        'interpolated_x': np.linspace(min(x_pts), max(x_pts), 100).tolist(),
        'interpolated_y': np.poly1d(expanded_poly_coeffs_np)(np.linspace(min(x_pts), max(x_pts), 100)).tolist(),
    })
    headers = ['n', 'x_i', 'y = f[xi]'] + [str(i) for i in range(1, n)]
    display_table = [[i, x_pts[i]] + [dd_table[i, j] if i <= n - 1 - j else 0.0 for j in range(n)] for i in range(n)]


    # --- AÑADIDO: SYMPY y LATEX ---
    poly_sympy = sum(c * x_sym**(len(expanded_poly_coeffs_np) - 1 - i) for i, c in enumerate(expanded_poly_coeffs_np))
    poly_str_latex = sp.latex(poly_sympy.evalf(6), mul_symbol=' ')
    # --- FIN AÑADIDO ---

    return headers, display_table, newton_coeffs_divided_differences.tolist(), \
           poly_str_expanded, poly_str_newton_native, plot_data, \
           poly_sympy, poly_str_latex # <-- NUEVOS VALORES




def calculate_lagrange(points):
    """ Calcula Lagrange y devuelve datos + objeto Sympy + LaTeX. """
    # ... (tu código actual de lagrange, incluyendo total_expanded_poly_coeffs) ...
    n = len(points)
    x_pts = np.array([p['x'] for p in points], dtype=float)
    y_pts = np.array([p['y'] for p in points], dtype=float)
    # ... (Validaciones) ...
    # ... (Cálculo de lagrange_basis_for_table, poly_str_lagrange_native) ...
    total_expanded_poly_coeffs = np.array([0.0])
    for i_loop_idx in range(n):
        li_numerator_poly = np.array([1.0])
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx: continue
            li_numerator_poly = np.polymul(li_numerator_poly, np.array([1.0, -x_pts[j_loop_idx]]))
        li_denominator_value = 1.0
        for j_loop_idx in range(n):
            if i_loop_idx == j_loop_idx: continue
            li_denominator_value *= (x_pts[i_loop_idx] - x_pts[j_loop_idx])
        li_coeffs = li_numerator_poly / li_denominator_value
        term_coeffs = np.polymul(np.array([y_pts[i_loop_idx]]), li_coeffs)
        total_expanded_poly_coeffs = np.polyadd(total_expanded_poly_coeffs, term_coeffs)
    poly_str_expanded = format_polynomial(total_expanded_poly_coeffs.tolist())
    # ... (Cálculo de plot_data) ...
    plot_data = json.dumps({ # Simplified plot data calc
        'original_x': x_pts.tolist(), 'original_y': y_pts.tolist(),
        'interpolated_x': np.linspace(min(x_pts), max(x_pts), 100).tolist(),
        'interpolated_y': np.poly1d(total_expanded_poly_coeffs)(np.linspace(min(x_pts), max(x_pts), 100)).tolist(),
    })
    lagrange_basis_for_table = [] # Placeholder, asume que lo calculas
    poly_str_lagrange_native = "" # Placeholder

    # --- AÑADIDO: SYMPY y LATEX ---
    poly_sympy = sum(c * x_sym**(len(total_expanded_poly_coeffs) - 1 - i) for i, c in enumerate(total_expanded_poly_coeffs))
    poly_str_latex = sp.latex(poly_sympy.evalf(6), mul_symbol=' ')
    # --- FIN AÑADIDO ---

    return lagrange_basis_for_table, poly_str_lagrange_native, poly_str_expanded, plot_data, \
           poly_sympy, poly_str_latex # <-- NUEVOS VALORES





def calculate_linear_spline(points_data):
    """ Calcula Spline Lineal y devuelve datos + lista Sympy + string LaTeX. """
    points_data.sort(key=lambda p: p['x'])
    x = np.array([p['x'] for p in points_data], dtype=float)
    y = np.array([p['y'] for p in points_data], dtype=float)
    n = len(x)
    # ... (Validaciones) ...
    coeffs_table_list = []
    tracers_list = []
    plot_x_all = []
    plot_y_all = []
    # --- AÑADIDO: SYMPY y LATEX ---
    spline_sympy_list = []
    tracers_latex_list = []
    # --- FIN AÑADIDO ---

    for i in range(n - 1):
        xi, yi = x[i], y[i]
        xi1, yi1 = x[i+1], y[i+1]
        m_i = (yi1 - yi) / (xi1 - xi)
        c_i = yi - m_i * xi
        coeffs_table_list.append({'i': i, 'coeff1': m_i, 'coeff2': c_i})
        tracer_str_normal = f"{m_i:.6f}x {'+' if c_i >= 0 else '-'} {abs(c_i):.6f}"
        tracers_list.append({'i': i, 'tracer_str': tracer_str_normal})
        # --- AÑADIDO: SYMPY y LATEX ---
        Si = m_i * x_sym + c_i
        spline_sympy_list.append((Si, xi, xi1))
        tracers_latex_list.append(f"S_{{{i}}}(x) = {sp.latex(Si.evalf(6))}")
        # --- FIN AÑADIDO ---
        x_segment = np.linspace(xi, xi1, 50)
        y_segment = m_i * x_segment + c_i
        plot_x_all.extend(x_segment.tolist())
        plot_y_all.extend(y_segment.tolist())

    plot_data = json.dumps({
        'original_x': x.tolist(), 'original_y': y.tolist(),
        'interpolated_x': plot_x_all, 'interpolated_y': plot_y_all, 'type': 'spline'
    })
    
    poly_str_latex = "; \\; ".join(tracers_latex_list) # Un string representativo

    return coeffs_table_list, tracers_list, plot_data, \
           spline_sympy_list, poly_str_latex # <-- NUEVOS VALORES

def calculate_cubic_spline(points_data):
    """ Calcula Spline Cúbico y devuelve datos + lista Sympy + string LaTeX. """
    points_data.sort(key=lambda p: p['x'])
    x = np.array([p['x'] for p in points_data], dtype=float)
    y = np.array([p['y'] for p in points_data], dtype=float)
    n = len(x)
    # ... (Validaciones, cálculo de h, A, B, M, a, b, c, d) ...
    h = x[1:] - x[:-1]
    A = np.zeros((n - 2, n - 2))
    B = np.zeros(n - 2)
    for i in range(n - 2):
        A[i, i] = 2 * (h[i] + h[i+1])
        B[i] = 6 * ((y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i])
        if i > 0: A[i, i-1] = h[i]
        if i < n - 3: A[i, i+1] = h[i+1]
    M_internal = np.linalg.solve(A, B)
    M = np.concatenate(([0], M_internal, [0]))
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

    for i in range(n - 1):
        Si = (a[i] * (x_sym - x[i])**3 + b[i] * (x_sym - x[i])**2 + c[i] * (x_sym - x[i]) + d[i])
        Si_expanded = sp.expand(Si)
        poly = sp.Poly(Si_expanded, x_sym)
        A_i = float(poly.coeff_monomial(x_sym**3))
        B_i = float(poly.coeff_monomial(x_sym**2))
        C_i = float(poly.coeff_monomial(x_sym**1))
        D_i = float(poly.coeff_monomial(x_sym**0))
        coeffs_table_list.append({'i': i, 'coeff1': A_i, 'coeff2': B_i, 'coeff3': C_i, 'coeff4': D_i})
        tracer_str_latex = sp.latex(Si_expanded.evalf(6), mul_symbol=' ')
        tracers_list.append({'i': i, 'tracer_str': tracer_str_latex})
        # --- AÑADIDO: SYMPY y LATEX ---
        spline_sympy_list.append((Si_expanded, x[i], x[i+1]))
        tracers_latex_list.append(f"S_{{{i}}}(x) = {tracer_str_latex}")
        # --- FIN AÑADIDO ---
        x_segment = np.linspace(x[i], x[i+1], 50)
        func_Si = sp.lambdify(x_sym, Si_expanded, 'numpy')
        y_segment = func_Si(x_segment)
        plot_x_all.extend(x_segment.tolist())
        plot_y_all.extend(y_segment.tolist())

    plot_data = json.dumps({
        'original_x': x.tolist(), 'original_y': y.tolist(),
        'interpolated_x': plot_x_all, 'interpolated_y': plot_y_all, 'type': 'spline'
    })

    poly_str_latex = "; \\; ".join(tracers_latex_list) # Un string representativo

    return coeffs_table_list, tracers_list, plot_data, \
           spline_sympy_list, poly_str_latex # <-- NUEVOS VALORES