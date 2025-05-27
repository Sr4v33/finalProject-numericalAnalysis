# chapterThree/views.py (o un archivo dedicado como chapterThree/methods.py)

import numpy as np
import json

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