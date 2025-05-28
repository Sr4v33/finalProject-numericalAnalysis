import numpy as np
from core.utils.functions import parse_function, get_derivatives, display_table


# ==============================================================================
# --- Bisección ---
# ==============================================================================

def biseccion(f, xi, xs, tol, niter):
    resultados = []
    fi = f(xi)
    fs = f(xs)

    if fi == 0: return [], f'{xi} es raíz.'
    if fs == 0: return [], f'{xs} es raíz.'
    if fi * fs > 0: return [], 'El intervalo no es válido.'

    c = 0
    xm = (xi + xs) / 2.0
    fm = f(xm)
    error = abs(xs - xi) / 2.0 # Error inicial

    resultados.append({'Iteración': c, 'xi': xi, 'xs': xs, 'xm': xm, 'f(xm)': fm, 'Error': error})

    while error > tol and fm != 0 and c < niter:
        if fi * fm < 0:
            xs = xm
            fs = fm
        else:
            xi = xm
            fi = fm

        xa = xm
        xm = (xi + xs) / 2.0
        fm = f(xm)
        error = abs(xm - xa) # Usamos error absoluto entre iteraciones
        c += 1
        resultados.append({'Iteración': c, 'xi': xi, 'xs': xs, 'xm': xm, 'f(xm)': fm, 'Error': error})

    if fm == 0: mensaje = f'{xm:.10f} es raíz.'
    elif error < tol: mensaje = f'{xm:.10f} es aprox. raíz con tol={tol:.1e}'
    else: mensaje = f'Fracasó en {niter} iteraciones.'
    return resultados, mensaje


# ==============================================================================
# --- Regla Falsa ---
# ==============================================================================

def regla_falsa(f, xi, xs, tol, niter):
    resultados = []
    fi = f(xi)
    fs = f(xs)

    if fi == 0: return [], f'{xi} es raíz.'
    if fs == 0: return [], f'{xs} es raíz.'
    if fi * fs > 0: return [], 'El intervalo no es válido.'

    c = 0
    error = tol + 1

    if (fs - fi) == 0: return [], "División por cero (fs - fi = 0)."
    xm = xs - (fs * (xs - xi)) / (fs - fi)
    fm = f(xm)
    resultados.append({'Iteración': c, 'xi': xi, 'xs': xs, 'xm': xm, 'f(xm)': fm, 'Error': np.nan})

    while error > tol and fm != 0 and c < niter:
        xa = xm
        if fi * fm < 0: xs = xm; fs = fm
        else: xi = xm; fi = fm
        if (fs - fi) == 0: return resultados, f"División por cero en iteración {c+1}."
        xm = xs - (fs * (xs - xi)) / (fs - fi)
        fm = f(xm)
        error = abs(xm - xa)
        c += 1
        resultados.append({'Iteración': c, 'xi': xi, 'xs': xs, 'xm': xm, 'f(xm)': fm, 'Error': error})

    if fm == 0: mensaje = f'{xm:.10f} es raíz.'
    elif error < tol: mensaje = f'{xm:.10f} es aprox. raíz con tol={tol:.1e}'
    else: mensaje = f'Fracasó en {niter} iteraciones.'
    return resultados, mensaje


# ==============================================================================
# --- Punto Fijo ---
# ==============================================================================

def punto_fijo(f, g, x0, tol, niter):
    resultados = []
    c = 0
    error = tol + 1
    xn = x0
    gn = g(xn) 
    f_x = f(xn)
    

    resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': gn, 'f(xm)': f_x, 'Error': np.nan})
    
    while error > tol and c < niter:
        try:
            x0 = xn  
            xn = g(x0)  
            c += 1  
            f_xn = f(xn)  
            error = abs(x0 - xn)  
            

            resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': g(xn), 'f(xm)': f_xn, 'Error': error})
        
        except Exception as e:
            return resultados, f"Error al evaluar g(x): {e}"
    
    if error < tol:
        mensaje = f'{xn:.10f} es aprox. raíz con tol={tol:.1e}'
    else:
        mensaje = f'Fracasó en {niter} iteraciones.'
    
    return resultados, mensaje




# ==============================================================================
# --- Newton  ---
# ==============================================================================

def newton(f, f_prime, x0, tol, niter):
    resultados = []
    c = 0
    error = tol + 1
    xn = x0
    fn = f(xn)
    resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': np.nan, 'f(xm)': fn, 'Error': np.nan})
    while error > tol and fn != 0 and c < niter:
        fpn = f_prime(xn)
        if fpn == 0: return resultados, f"División por cero (f'({xn}) = 0)."
        xn_nuevo = xn - fn / fpn
        error = abs(xn_nuevo - xn)
        xn = xn_nuevo
        fn = f(xn)
        c += 1
        resultados.append({'Iteración': c, 'xi': xn, 'xs': fpn, 'xm': xn_nuevo, 'f(xm)': fn, 'Error': error})
    if fn == 0: mensaje = f'{xn:.10f} es aprox. raíz con tol={tol:.1e}.'
    elif error < tol: mensaje = f'{xn:.10f} es aprox. raíz con tol={tol:.1e}'
    else: mensaje = f'Fracasó en {niter} iteraciones.'
    return resultados, mensaje


# ==============================================================================
# --- Secante ---
# ==============================================================================

import numpy as np

def secante(f, x0, x1, tol, niter):
    """
    Implementa el método de la secante para encontrar una raíz de la función f.

    Args:
        f (function): La función para la cual se busca la raíz.
        x0 (float): Primera aproximación inicial.
        x1 (float): Segunda aproximación inicial.
        tol (float): La tolerancia para el criterio de parada (error).
        niter (int): El número máximo de iteraciones permitidas.

    Returns:
        tuple: Una tupla que contiene:
            - list: Una lista de diccionarios, donde cada diccionario representa
                    una iteración con las claves 'Iteration (i)', 'xi', 'f(xi)', 'E'.
            - str: Un mensaje indicando el resultado del método.
    """
    resultados = []
    c = 0  # Contador de iteraciones

    fx0 = f(x0)
    # Añadir x0 (Iteración 0)
    resultados.append({'Iteración': c, 'xi': x0, 'xs': np.nan, 'xm': np.nan, 'f(xm)': fx0, 'Error': np.nan})

    if fx0 == 0:
        return resultados, f"{x0:.10f} es una raíz."

    if x0 == x1:
        return resultados, f"Error: x0 y x1 no pueden ser iguales (x0 = {x0}, x1 = {x1})."

    c += 1
    fx1 = f(x1)
    # Añadir x1 (Iteración 1)
    # El error E = |x1 - x0| podría ir aquí, pero seguimos la imagen 
    # donde el primer error aparece en la iteración 2.
    resultados.append({'Iteración': c, 'xi': x1, 'xs': x0, 'xm': np.nan, 'f(xm)': fx1, 'Error': np.nan}) 

    if fx1 == 0:
        return resultados, f"{x1:.10f} es una raíz."

    error = tol + 1  # Inicializar error para entrar al bucle

    # Bucle principal del método de la secante
    while error > tol and fx1 != 0 and c < niter:
        denominador = fx1 - fx0
        
        if denominador == 0:
            return resultados, f"División por cero (f({x1:.10f}) - f({x0:.10f}) = 0). El método no puede continuar."

        x_nuevo = x1 - (fx1 * (x1 - x0)) / denominador
        error = abs(x_nuevo - x1)  # Error: |xi - x(i-1)|
        
        x0 = x1
        fx0 = fx1
        x1 = x_nuevo
        fx1 = f(x1)
        
        c += 1
        
        # Añadir la iteración actual a los resultados
        resultados.append({'Iteración': c, 'xi': x1, 'xs': x0, 'xm': x_nuevo, 'f(xm)': fx1, 'Error': error})

    # Determinar el mensaje final
    if fx1 == 0:
        mensaje = f'{x1:.10f} es una raíz.'
    elif error < tol:
        mensaje = f'{x1:.10f} es una aproximación a la raíz con tol={tol:.1e}'
    else:
        mensaje = f'Fracasó en {niter} iteraciones.'
        
    return resultados, mensaje


# ==============================================================================
# --- Newton Modificado ---
# ==============================================================================

def newton_modificado(f, f_prime, f_double_prime, x0, tol, niter):

    resultados = []
    xn = float(x0) 
    
    try:
        fn = f(xn)
        fpn = f_prime(xn)
        fppn = f_double_prime(xn)
    except Exception as e:
        return [], f"Error al evaluar la función o derivadas en x0={x0}: {e}"

    c = 0
    error = tol + 1
    
    # Añadir la iteración inicial (0)
    resultados.append({
        'Iteración': c, 
        'xi': xn, 
        'xs': np.nan,  
        'xm': np.nan,  
        'f(xm)': fn,  # Mostramos f(x0)
        'Error': np.nan # Sin error aún
    })

    while error > tol and fn != 0 and c < niter:
        denominador = fpn**2 - fn * fppn

        
        if abs(denominador) < 1e-15: # Usamos un umbral pequeño en lugar de 0 exacto
            return resultados, f"El método falló: Denominador cercano a cero en x={xn:.10f} (Iteración {c+1}). Posible raíz múltiple o divergencia."

        xn_nuevo = xn - (fn * fpn) / denominador
        
        
        try:
            fn_nuevo = f(xn_nuevo)
            fpn_nuevo = f_prime(xn_nuevo)
            fppn_nuevo = f_double_prime(xn_nuevo)
        except Exception as e:
            return resultados, f"Error al evaluar en x={xn_nuevo:.10f} (Iteración {c+1}): {e}. El método se detiene."

        error = abs(xn_nuevo - xn)
        
        xn = xn_nuevo
        fn = fn_nuevo
        fpn = fpn_nuevo
        fppn = fppn_nuevo
        
        c += 1

        resultados.append({
            'Iteración': c, 
            'xi': xn, 
            'xs': fpn, 
            'xm': xn_nuevo, 
            'f(xm)': fn, 
            'Error': error
        })


    if fn == 0:
        mensaje = f'x = {xn:.10f} es raíz.'
    elif error <= tol:
        mensaje = f'{xn:.10f} es aprox. raíz con tol={tol:.1e}'
    elif c == niter:
        mensaje = f'Se alcanzó el número máximo de {niter} iteraciones. No se encontró raíz con la tolerancia dada.'
    else:
        # Esta condición es menos probable con la verificación del denominador, pero la mantenemos.
        mensaje = 'El método fracasó o explotó antes de alcanzar la convergencia o el máximo de iteraciones.'
        
    return resultados, mensaje