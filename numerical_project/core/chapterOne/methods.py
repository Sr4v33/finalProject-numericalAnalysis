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

def punto_fijo(g, x0, tol, niter):
    resultados = []
    c = 0
    error = tol + 1
    xn = x0
    resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': np.nan, 'f(xm)': np.nan, 'Error': np.nan})
    while error > tol and c < niter:
        try:
            xn_nuevo = g(xn)
            error = abs(xn_nuevo - xn)
            c += 1
            xn = xn_nuevo
            resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': g(xn), 'f(xm)': np.nan, 'Error': error})
        except Exception as e: return resultados, f"Error al evaluar g(x): {e}"
    if error < tol: mensaje = f'{xn:.10f} es aprox. punto fijo con tol={tol:.1e}'
    else: mensaje = f'Fracasó en {niter} iteraciones.'
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
    if fn == 0: mensaje = f'{xn:.10f} es raíz.'
    elif error < tol: mensaje = f'{xn:.10f} es aprox. raíz con tol={tol:.1e}'
    else: mensaje = f'Fracasó en {niter} iteraciones.'
    return resultados, mensaje


# ==============================================================================
# --- Secante ---
# ==============================================================================

def secante(f, x0, x1, tol, niter):
    resultados = []
    c = 0
    error = tol + 1
    f0 = f(x0)
    if f0 == 0: return [], f'{x0} es raíz.'
    f1 = f(x1)
    resultados.append({'Iteración': c, 'xi': x0, 'xs': x1, 'xm': x1, 'f(xm)': f1, 'Error': np.nan})
    while error > tol and f1 != 0 and c < niter:
        if (f1 - f0) == 0: return resultados, f"División por cero (f(x1) - f(x0) = 0)."
        x_nuevo = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x_nuevo - x1)
        x0 = x1; f0 = f1; x1 = x_nuevo; f1 = f(x1); c += 1
        resultados.append({'Iteración': c, 'xi': x0, 'xs': x1, 'xm': x_nuevo, 'f(xm)': f1, 'Error': error})
    if f1 == 0: mensaje = f'{x1:.10f} es raíz.'
    elif error < tol: mensaje = f'{x1:.10f} es aprox. raíz con tol={tol:.1e}'
    else: mensaje = f'Fracasó en {niter} iteraciones.'
    return resultados, mensaje


# ==============================================================================
# --- Newton Modificado ---
# ==============================================================================

def newton_modificado(f, f_prime, f_double_prime, x0, tol, niter):
    resultados = []
    c = 0
    error = tol + 1
    xn = x0
    fn = f(xn)
    resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': np.nan, 'f(xm)': fn, 'Error': np.nan})
    while error > tol and fn != 0 and c < niter:
        fpn = f_prime(xn)
        fppn = f_double_prime(xn)
        denominador = fpn**2 - fn * fppn
        if denominador == 0: return resultados, f"División por cero (Denominador = 0)."
        xn_nuevo = xn - (fn * fpn) / denominador
        error = abs(xn_nuevo - xn)
        xn = xn_nuevo
        fn = f(xn)
        c += 1
        resultados.append({'Iteración': c, 'xi': xn, 'xs': np.nan, 'xm': xn_nuevo, 'f(xm)': fn, 'Error': error})
    if fn == 0: mensaje = f'{xn:.10f} es raíz.'
    elif error < tol: mensaje = f'{xn:.10f} es aprox. raíz con tol={tol:.1e}'
    else: mensaje = f'Fracasó en {niter} iteraciones.'
    return resultados, mensaje