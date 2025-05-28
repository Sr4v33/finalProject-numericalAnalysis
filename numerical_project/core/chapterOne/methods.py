import numpy as np
# Se asume que este módulo contiene las funciones parse_function, get_derivatives, display_table
# from core.utils.functions import parse_function, get_derivatives, display_table

# ==============================================================================
# --- Bisección ---
# ==============================================================================

def biseccion(f, xi, xs, tol, niter):
    """
    Encuentra una raíz de la función 'f' en el intervalo [xi, xs] usando el
    método de bisección.

    El método de bisección es un algoritmo de búsqueda incremental que divide
    repetidamente un intervalo por la mitad y selecciona el subintervalo en
    el cual debe existir una raíz.

    Args:
        f (callable): La función para la cual se busca la raíz. Debe aceptar
                      un argumento numérico y retornar un valor numérico.
        xi (float): El extremo inferior del intervalo inicial.
        xs (float): El extremo superior del intervalo inicial.
        tol (float): La tolerancia deseada para el error. El algoritmo se
                     detiene cuando el error absoluto es menor que 'tol'.
        niter (int): El número máximo de iteraciones permitidas.

    Returns:
        tuple: Una tupla que contiene:
            - list: Una lista de diccionarios, donde cada diccionario
                    representa una iteración y contiene 'Iteración', 'xi',
                    'xs', 'xm', 'f(xm)' y 'Error'.
            - str: Un mensaje indicando el resultado del método (si se
                   encontró una raíz, si se alcanzó 'niter', o si el
                   intervalo inicial no es válido).
    """
    resultados = []  # Lista para almacenar los resultados de cada iteración
    fi = f(xi)       # Evaluar la función en el extremo inferior
    fs = f(xs)       # Evaluar la función en el extremo superior

    # --- Verificaciones Iniciales ---
    if fi == 0:
        return [], f'{xi} es raíz.'
    if fs == 0:
        return [], f'{xs} es raíz.'
    if fi * fs > 0:
        return [], 'El intervalo no es válido (f(xi) * f(xs) > 0).'

    # --- Inicialización del Método ---
    c = 0                     # Contador de iteraciones
    xm = (xi + xs) / 2.0      # Primera aproximación (punto medio)
    fm = f(xm)                # Evaluar la función en el punto medio
    error = abs(xs - xi) / 2.0  # Error inicial (amplitud / 2)

    # Almacenar la primera iteración (iteración 0)
    resultados.append({'Iteración': c, 'xi': xi, 'xs': xs, 'xm': xm, 'f(xm)': fm, 'Error': error})

    # --- Bucle Principal de Bisección ---
    while error > tol and fm != 0 and c < niter:
        # Determinar el nuevo subintervalo
        if fi * fm < 0:
            xs = xm  # La raíz está en [xi, xm]
            fs = fm
        else:
            xi = xm  # La raíz está en [xm, xs]
            fi = fm

        xa = xm  # Guardar el xm anterior para calcular el error
        xm = (xi + xs) / 2.0  # Calcular el nuevo punto medio
        fm = f(xm)            # Evaluar la función en el nuevo punto medio
        error = abs(xm - xa)  # Calcular el error absoluto
        c += 1                # Incrementar el contador de iteraciones

        # Almacenar los resultados de la iteración actual
        resultados.append({'Iteración': c, 'xi': xi, 'xs': xs, 'xm': xm, 'f(xm)': fm, 'Error': error})

    # --- Determinación del Mensaje Final ---
    if fm == 0:
        mensaje = f'{xm:.10f} es raíz.'
    elif error < tol:
        mensaje = f'{xm:.10f} es una aproximación a la raíz con tolerancia = {tol:.1e}'
    else:
        mensaje = f'Fracasó en {niter} iteraciones. No se encontró la raíz con la tolerancia dada.'

    return resultados, mensaje


# ==============================================================================
# --- Regla Falsa ---
# ==============================================================================

def regla_falsa(f, xi, xs, tol, niter):
    """
    Encuentra una raíz de la función 'f' en el intervalo [xi, xs] usando el
    método de la Regla Falsa (Falsa Posición).

    Similar a la bisección, pero calcula el punto intermedio (xm) como la
    intersección de la recta que une (xi, f(xi)) y (xs, f(xs)) con el eje x,
    lo que generalmente acelera la convergencia (aunque a veces puede estancarse).

    Args:
        f (callable): La función para la cual se busca la raíz.
        xi (float): El extremo inferior del intervalo inicial.
        xs (float): El extremo superior del intervalo inicial.
        tol (float): La tolerancia deseada para el error.
        niter (int): El número máximo de iteraciones.

    Returns:
        tuple: Una tupla que contiene:
            - list: Una lista de diccionarios con los resultados de cada iteración.
            - str: Un mensaje indicando el resultado del método.
    """
    resultados = []
    fi = f(xi)
    fs = f(xs)

    # --- Verificaciones Iniciales ---
    if fi == 0: return [], f'{xi} es raíz.'
    if fs == 0: return [], f'{xs} es raíz.'
    if fi * fs > 0: return [], 'El intervalo no es válido.'

    # --- Inicialización del Método ---
    c = 0
    error = tol + 1  # Inicializar error para entrar al bucle

    # Verificar posible división por cero inicial
    if (fs - fi) == 0:
        return [], "División por cero inicial (fs - fi = 0). No se puede aplicar Regla Falsa."

    # Calcular el primer xm usando la fórmula de Regla Falsa
    xm = xs - (fs * (xs - xi)) / (fs - fi)
    fm = f(xm)
    # Almacenar la primera iteración (Error no aplica aún)
    resultados.append({'Iteración': c, 'xi': xi, 'xs': xs, 'xm': xm, 'f(xm)': fm, 'Error': np.nan})

    # --- Bucle Principal de Regla Falsa ---
    while error > tol and fm != 0 and c < niter:
        xa = xm  # Guardar xm anterior

        # Actualizar el intervalo
        if fi * fm < 0:
            xs = xm
            fs = fm
        else:
            xi = xm
            fi = fm

        # Verificar posible división por cero en la iteración
        if (fs - fi) == 0:
            return resultados, f"División por cero en iteración {c+1}. El método se detiene."

        # Calcular el nuevo xm
        xm = xs - (fs * (xs - xi)) / (fs - fi)
        fm = f(xm)
        error = abs(xm - xa)  # Calcular el error
        c += 1

        # Almacenar resultados
        resultados.append({'Iteración': c, 'xi': xi, 'xs': xs, 'xm': xm, 'f(xm)': fm, 'Error': error})

    # --- Determinación del Mensaje Final ---
    if fm == 0:
        mensaje = f'{xm:.10f} es raíz.'
    elif error < tol:
        mensaje = f'{xm:.10f} es una aproximación a la raíz con tol = {tol:.1e}'
    else:
        mensaje = f'Fracasó en {niter} iteraciones.'

    return resultados, mensaje


# ==============================================================================
# --- Punto Fijo ---
# ==============================================================================

def punto_fijo(f, g, x0, tol, niter):
    """
    Encuentra una raíz de la función 'f' usando el método de iteración de
    Punto Fijo, basado en una función de iteración 'g(x)' tal que x = g(x).

    El método genera una secuencia x_n+1 = g(x_n) que, si converge, lo hace
    hacia un punto fijo de 'g', que idealmente es una raíz de 'f'.
    La convergencia depende de la elección de 'g(x)' y 'x0'.

    Args:
        f (callable): La función original f(x) para la cual se busca la raíz
                      (usada principalmente para evaluar f(xm) en los resultados).
        g (callable): La función de iteración g(x).
        x0 (float): La aproximación inicial.
        tol (float): La tolerancia deseada para el error.
        niter (int): El número máximo de iteraciones.

    Returns:
        tuple: Una tupla que contiene:
            - list: Una lista de diccionarios con los resultados de cada iteración.
                    ('xm' representa g(xi) y 'f(xm)' representa f(xi) de la tabla).
            - str: Un mensaje indicando el resultado del método.
    """
    resultados = []
    c = 0
    error = tol + 1
    xn = x0  # Valor actual (xi en la tabla)

    try:
        gn = g(xn)  # g(xi) (xm en la tabla)
        f_x = f(xn) # f(xi) (f(xm) en la tabla)
    except Exception as e:
        return [], f"Error al evaluar f(x) o g(x) en x0={x0}: {e}"

    # Almacenar la iteración inicial (0)
    resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': gn, 'f(xm)': f_x, 'Error': np.nan})

    # --- Bucle Principal de Punto Fijo ---
    while error > tol and c < niter:
        try:
            x_prev = xn     # Guardar el valor anterior
            xn = g(x_prev)  # Calcular el nuevo valor: x_n+1 = g(x_n)
            f_xn = f(xn)    # Evaluar f en el nuevo valor
            error = abs(xn - x_prev) # Calcular el error
            c += 1          # Incrementar contador

            # Almacenar resultados
            resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': g(xn), 'f(xm)': f_xn, 'Error': error})

        except Exception as e:
            # Capturar errores durante la evaluación (p.ej., división por cero, log de negativo)
            return resultados, f"Error al evaluar g(x) en iteración {c+1}: {e}. El método se detiene."

    # --- Determinación del Mensaje Final ---
    if error < tol:
        mensaje = f'{xn:.10f} es una aproximación al punto fijo (raíz) con tol = {tol:.1e}'
    else:
        mensaje = f'Fracasó en {niter} iteraciones.'

    return resultados, mensaje


# ==============================================================================
# --- Newton-Raphson ---
# ==============================================================================

def newton(f, f_prime, x0, tol, niter):
    """
    Encuentra una raíz de la función 'f' usando el método de Newton-Raphson.

    Este método utiliza la tangente a la curva en el punto actual para
    estimar la siguiente aproximación a la raíz. La fórmula de iteración es:
    x_n+1 = x_n - f(x_n) / f'(x_n).

    Args:
        f (callable): La función para la cual se busca la raíz.
        f_prime (callable): La primera derivada de la función 'f'.
        x0 (float): La aproximación inicial.
        tol (float): La tolerancia deseada para el error.
        niter (int): El número máximo de iteraciones.

    Returns:
        tuple: Una tupla que contiene:
            - list: Una lista de diccionarios con los resultados de cada iteración.
                    ('xi' es xn, 'f(xm)' es f(xn), 'xs' es f'(xn), 'xm' es xn_nuevo).
            - str: Un mensaje indicando el resultado del método.
    """
    resultados = []
    c = 0
    error = tol + 1
    xn = x0
    fn = f(xn)

    # Almacenar la iteración inicial (0)
    resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': np.nan, 'f(xm)': fn, 'Error': np.nan})

    # --- Bucle Principal de Newton ---
    while error > tol and fn != 0 and c < niter:
        fpn = f_prime(xn)  # Calcular la derivada en el punto actual

        # Verificar división por cero (tangente horizontal)
        if fpn == 0:
            return resultados, f"División por cero (f'({xn:.10f}) = 0). El método se detiene."

        xn_nuevo = xn - fn / fpn  # Fórmula de Newton
        error = abs(xn_nuevo - xn)  # Calcular error
        xn = xn_nuevo             # Actualizar el punto
        fn = f(xn)                # Evaluar f en el nuevo punto
        c += 1                    # Incrementar contador

        # Almacenar resultados
        resultados.append({'Iteración': c, 'xi': xn, 'xs': fpn, 'xm': xn_nuevo, 'f(xm)': fn, 'Error': error})

    # --- Determinación del Mensaje Final ---
    if fn == 0:
        mensaje = f'{xn:.10f} es raíz.'
    elif error < tol:
        mensaje = f'{xn:.10f} es una aproximación a la raíz con tol = {tol:.1e}'
    else:
        mensaje = f'Fracasó en {niter} iteraciones.'

    return resultados, mensaje


# ==============================================================================
# --- Secante ---
# ==============================================================================

def secante(f, x0, x1, tol, niter):
    """
    Encuentra una raíz de la función 'f' usando el método de la Secante.

    Similar a Newton, pero aproxima la derivada usando una diferencia finita
    basada en los dos últimos puntos. Usa la recta secante en lugar de la
    tangente. Fórmula: x_n+1 = x_n - f(x_n) * (x_n - x_n-1) / (f(x_n) - f(x_n-1)).

    Args:
        f (callable): La función para la cual se busca la raíz.
        x0 (float): Primera aproximación inicial.
        x1 (float): Segunda aproximación inicial.
        tol (float): La tolerancia deseada para el error.
        niter (int): El número máximo de iteraciones.

    Returns:
        tuple: Una tupla que contiene:
            - list: Una lista de diccionarios con los resultados de cada iteración.
                    ('xi' es x1, 'xs' es x0, 'xm' es x_nuevo, 'f(xm)' es fx1).
            - str: Un mensaje indicando el resultado del método.
    """
    resultados = []
    c = 0  # Contador

    fx0 = f(x0)
    # Almacenar x0 (Iteración 0)
    resultados.append({'Iteración': c, 'xi': x0, 'xs': np.nan, 'xm': np.nan, 'f(xm)': fx0, 'Error': np.nan})

    if fx0 == 0:
        return resultados, f"{x0:.10f} es una raíz."

    # Verificar que x0 y x1 no sean iguales
    if x0 == x1:
        return resultados, f"Error: x0 y x1 no pueden ser iguales (x0 = {x0}, x1 = {x1})."

    c += 1
    fx1 = f(x1)
    # Almacenar x1 (Iteración 1)
    resultados.append({'Iteración': c, 'xi': x1, 'xs': x0, 'xm': np.nan, 'f(xm)': fx1, 'Error': abs(x1 - x0)}) # Primer error

    if fx1 == 0:
        return resultados, f"{x1:.10f} es una raíz."

    error = tol + 1  # Inicializar error

    # --- Bucle Principal de Secante ---
    while error > tol and fx1 != 0 and c < niter:
        denominador = fx1 - fx0

        # Verificar división por cero
        if denominador == 0:
            return resultados, f"División por cero (f({x1:.10f}) - f({x0:.10f}) = 0). El método no puede continuar."

        # Fórmula de la Secante
        x_nuevo = x1 - (fx1 * (x1 - x0)) / denominador
        error = abs(x_nuevo - x1)  # Calcular error

        # Actualizar puntos para la siguiente iteración
        x0 = x1
        fx0 = fx1
        x1 = x_nuevo
        fx1 = f(x1)

        c += 1

        # Almacenar resultados
        resultados.append({'Iteración': c, 'xi': x1, 'xs': x0, 'xm': x_nuevo, 'f(xm)': fx1, 'Error': error})

    # --- Determinación del Mensaje Final ---
    if fx1 == 0:
        mensaje = f'{x1:.10f} es una raíz.'
    elif error < tol:
        mensaje = f'{x1:.10f} es una aproximación a la raíz con tol = {tol:.1e}'
    else:
        mensaje = f'Fracasó en {niter} iteraciones.'

    return resultados, mensaje


# ==============================================================================
# --- Newton Modificado ---
# ==============================================================================

def newton_modificado(f, f_prime, f_double_prime, x0, tol, niter):
    """
    Encuentra una raíz de la función 'f' usando el método de Newton Modificado.

    Este método ajusta la fórmula de Newton para mejorar la convergencia,
    especialmente cerca de raíces múltiples. La fórmula de iteración es:
    x_n+1 = x_n - (f(x_n) * f'(x_n)) / ( (f'(x_n))^2 - f(x_n) * f''(x_n) ).

    Args:
        f (callable): La función para la cual se busca la raíz.
        f_prime (callable): La primera derivada de 'f'.
        f_double_prime (callable): La segunda derivada de 'f'.
        x0 (float): La aproximación inicial.
        tol (float): La tolerancia deseada para el error.
        niter (int): El número máximo de iteraciones.

    Returns:
        tuple: Una tupla que contiene:
            - list: Una lista de diccionarios con los resultados de cada iteración.
                    ('xi' es xn, 'xs' es fpn, 'xm' es xn_nuevo, 'f(xm)' es fn).
            - str: Un mensaje indicando el resultado del método.
    """
    resultados = []
    xn = float(x0)  # Asegurar que x0 sea float

    # --- Evaluación Inicial y Verificación ---
    try:
        fn = f(xn)
        fpn = f_prime(xn)
        fppn = f_double_prime(xn)
    except Exception as e:
        return [], f"Error al evaluar la función o derivadas en x0={x0}: {e}"

    c = 0
    error = tol + 1

    # Almacenar la iteración inicial (0)
    resultados.append({'Iteración': c, 'xi': xn, 'xs': fpn, 'xm': np.nan, 'f(xm)': fn, 'Error': np.nan})

    # --- Bucle Principal de Newton Modificado ---
    while error > tol and fn != 0 and c < niter:
        # Calcular el denominador de la fórmula
        denominador = fpn**2 - fn * fppn

        # Verificar si el denominador es cercano a cero para evitar división por cero
        if abs(denominador) < 1e-15:  # Umbral pequeño
            return resultados, f"Denominador cercano a cero en x={xn:.10f} (Iteración {c+1}). El método se detiene."

        # Fórmula de Newton Modificado
        xn_nuevo = xn - (fn * fpn) / denominador

        # --- Evaluación y Actualización ---
        try:
            fn_nuevo = f(xn_nuevo)
            fpn_nuevo = f_prime(xn_nuevo)
            fppn_nuevo = f_double_prime(xn_nuevo)
        except Exception as e:
            return resultados, f"Error al evaluar en x={xn_nuevo:.10f} (Iteración {c+1}): {e}. El método se detiene."

        error = abs(xn_nuevo - xn) # Calcular error

        # Actualizar valores para la siguiente iteración
        xn = xn_nuevo
        fn = fn_nuevo
        fpn = fpn_nuevo
        fppn = fppn_nuevo

        c += 1

        # Almacenar resultados
        resultados.append({'Iteración': c, 'xi': xn, 'xs': fpn, 'xm': xn_nuevo, 'f(xm)': fn, 'Error': error})

    # --- Determinación del Mensaje Final ---
    if fn == 0:
        mensaje = f'{xn:.10f} es raíz.'
    elif error <= tol:
        mensaje = f'{xn:.10f} es una aproximación a la raíz con tol = {tol:.1e}'
    else: # c == niter
        mensaje = f'Fracasó en {niter} iteraciones.'

    return resultados, mensaje