import sympy
import numpy as np
import re

def parse_function(function_string, variable='x'):
    """
    Analiza una cadena de texto que representa una función matemática y la
    convierte en una función de Python que puede ser evaluada numéricamente.

    Utiliza la biblioteca SymPy para el análisis simbólico y `lambdify`
    para crear una función compatible con NumPy, permitiendo el uso de
    funciones matemáticas de NumPy (como exp, cos, sin, log, sqrt, etc.).

    Args:
        function_string (str): La cadena de texto que contiene la función.
                               Ejemplo: 'x**2 + sin(x)'.
        variable (str, optional): El nombre de la variable independiente
                                    en la función. Por defecto es 'x'.

    Returns:
        tuple: Una tupla que contiene:
            - function (callable): La función Python evaluable si el análisis
                                   es exitoso.
            - error_message (str or None): Un mensaje de error si ocurre
                                         un problema durante el análisis,
                                         o None si tiene éxito.
    """
    try:
        # Reemplaza el operador de potencia '^' por '**' (compatible con Python/SymPy)
        function_string = function_string.replace('^', '**')

        # Define el símbolo de la variable independiente usando SymPy
        x = sympy.symbols(variable)

        # Convierte la cadena de texto en una expresión simbólica de SymPy.
        # Esto valida la sintaxis y estructura la función.
        expr = sympy.sympify(function_string)

        # Convierte la expresión SymPy en una función Python evaluable.
        # 'lambdify' utiliza NumPy para una evaluación numérica eficiente.
        func = sympy.lambdify(x, expr, 'numpy')

        # Retorna la función creada y None (indicando que no hubo error)
        return func, None

    except (sympy.SympifyError, TypeError, SyntaxError) as e:
        # Captura excepciones comunes durante el análisis y retorna None
        # y un mensaje de error descriptivo.
        return None, f"Error al analizar la función '{function_string}': {e}"


def get_derivatives(function_string, variable='x'):
    """
    Analiza una cadena de texto de una función, calcula su primera y segunda
    derivada simbólicamente, y convierte la función original y sus
    derivadas en funciones Python evaluables.

    Utiliza SymPy para el análisis, diferenciación simbólica y `lambdify`
    para la conversión a funciones numéricas compatibles con NumPy.

    Args:
        function_string (str): La cadena de texto que contiene la función.
                               Ejemplo: 'exp(x) * cos(x)'.
        variable (str, optional): El nombre de la variable independiente.
                                    Por defecto es 'x'.

    Returns:
        tuple: Una tupla que contiene:
            - f (callable or None): La función original evaluable.
            - f_prime (callable or None): La primera derivada evaluable.
            - f_double_prime (callable or None): La segunda derivada evaluable.
            - error_message (str or None): Un mensaje de error si ocurre
                                         un problema, o None si tiene éxito.
    """
    try:
        # Reemplaza el operador de potencia '^' por '**'
        function_string = function_string.replace('^', '**')

        # Define el símbolo de la variable independiente
        x = sympy.symbols(variable)

        # Convierte la cadena en una expresión SymPy
        expr = sympy.sympify(function_string)

        # Calcula la primera derivada simbólica
        f_prime_expr = sympy.diff(expr, x)
        # Calcula la segunda derivada simbólica (derivada de la primera)
        f_double_prime_expr = sympy.diff(f_prime_expr, x)

        # Convierte la expresión original y sus derivadas en funciones Python
        f = sympy.lambdify(x, expr, 'numpy')
        f_prime = sympy.lambdify(x, f_prime_expr, 'numpy')
        f_double_prime = sympy.lambdify(x, f_double_prime_expr, 'numpy')

        # Retorna las tres funciones y None (sin error)
        return f, f_prime, f_double_prime, None

    except (sympy.SympifyError, TypeError, SyntaxError) as e:
        # Captura excepciones y retorna None para las funciones y un mensaje de error.
        return None, None, None, f"Error al derivar la función '{function_string}': {e}"


def display_table(results):
    """
    Genera una tabla HTML formateada a partir de una lista de diccionarios.

    Esta función está diseñada para mostrar resultados tabulares, como los
    generados por métodos numéricos, en un formato web. Maneja el formato
    de números de punto flotante y valores NaN (Not a Number).

    Nota: Aunque es funcional, en un entorno como Django, generalmente es
    preferible manejar la renderización de tablas directamente en las
    plantillas (templates) para una mejor separación de la lógica y la
    presentación.

    Args:
        results (list of dict): Una lista donde cada elemento es un
                                  diccionario que representa una fila de la
                                  tabla. Se asume que todos los diccionarios
                                  tienen las mismas claves, que se usarán
                                  como encabezados de columna.

    Returns:
        str: Una cadena de texto que contiene el código HTML de la tabla,
             o un mensaje si no hay resultados.
    """
    # Si la lista de resultados está vacía, retorna un mensaje indicándolo.
    if not results:
        return "<p>No hay resultados para mostrar.</p>"

    # Obtiene los encabezados de la tabla a partir de las claves del primer diccionario.
    headers = results[0].keys()

    # Inicia la construcción de la tabla HTML con clases de Bootstrap.
    html = '<table class="table table-bordered"><thead><tr>'
    # Agrega los encabezados a la tabla.
    for header in headers:
        html += f'<th>{header}</th>'
    html += '</tr></thead><tbody>'

    # Itera sobre cada fila (diccionario) en los resultados.
    for row in results:
        html += '<tr>'
        # Itera sobre cada encabezado para obtener el valor correspondiente en la fila.
        for header in headers:
            value = row.get(header) # Usa .get() para seguridad, aunque se asumen claves.

            # Verifica si el valor es un número de punto flotante.
            if isinstance(value, float):
                # Si es NaN, muestra '---'.
                if np.isnan(value):
                    html += '<td>---</td>'
                # Si es un float válido, lo formatea a 10 decimales.
                else:
                    html += f'<td>{value:.10f}</td>'
            # Si no es float, lo muestra tal cual.
            else:
                html += f'<td>{value}</td>'
        html += '</tr>'

    # Cierra las etiquetas del cuerpo y de la tabla.
    html += '</tbody></table>'
    return html