import sympy
import numpy as np
import re

def parse_function(function_string, variable='x'):
    """
    Parsea una cadena de texto y la convierte en una función Python evaluable.
    Permite usar funciones numpy como exp, cos, sin, log, etc.
    """
    try:
        # Reemplazar '^' por '**' si se usa para potencias
        function_string = function_string.replace('^', '**')

        # Definir el símbolo
        x = sympy.symbols(variable)
        

        # Convertir la cadena a una expresión SymPy
        expr = sympy.sympify(function_string)
        
        # Convertir la expresión SymPy a una función Python usando numpy
        # 'lambdify' crea una función rápida para evaluación numérica.
        func = sympy.lambdify(x, expr, 'numpy')
        
        return func, None # Retorna la función y None para indicar éxito

    except (sympy.SympifyError, TypeError, SyntaxError) as e:
        return None, f"Error al parsear la función '{function_string}': {e}"


def get_derivatives(function_string, variable='x'):
    """
    Parsea una función, calcula su primera y segunda derivada,
    y retorna las tres como funciones Python evaluables.
    """
    try:
        # Reemplazar '^' por '**' si se usa para potencias
        function_string = function_string.replace('^', '**')

        x = sympy.symbols(variable)
        
        # Convertir la cadena a una expresión SymPy
        expr = sympy.sympify(function_string)
        
        # Calcular derivadas
        f_prime_expr = sympy.diff(expr, x)
        f_double_prime_expr = sympy.diff(f_prime_expr, x)
        
        # Convertir a funciones Python
        f = sympy.lambdify(x, expr, 'numpy')
        f_prime = sympy.lambdify(x, f_prime_expr, 'numpy')
        f_double_prime = sympy.lambdify(x, f_double_prime_expr, 'numpy')
        
        return f, f_prime, f_double_prime, None

    except (sympy.SympifyError, TypeError, SyntaxError) as e:
        return None, None, None, f"Error al derivar la función '{function_string}': {e}"



def display_table(results):
    """
    (Opcional) Genera una tabla HTML a partir de los resultados.
    Aunque es mejor hacerlo directamente en la plantilla de Django.
    """
    if not results:
        return "<p>No hay resultados para mostrar.</p>"

    headers = results[0].keys()
    html = '<table class="table table-bordered"><thead><tr>'
    for header in headers:
        html += f'<th>{header}</th>'
    html += '</tr></thead><tbody>'

    for row in results:
        html += '<tr>'
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                # Formatear floats, manejar NaN
                if np.isnan(value):
                    html += '<td>---</td>'
                else:
                    html += f'<td>{value:.10f}</td>'
            else:
                 html += f'<td>{value}</td>'
        html += '</tr>'
    
    html += '</tbody></table>'
    return html