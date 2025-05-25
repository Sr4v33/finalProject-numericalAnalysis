from django.shortcuts import render
from .forms import NumericalMethodForm
from . import methods as chapter1_methods
from core.utils.functions import parse_function, get_derivatives
import numpy as np

def chapter_one_view(request):
    form = NumericalMethodForm()
    context = {'form': form}

    if request.method == 'POST':
        form = NumericalMethodForm(request.POST)
        context['form'] = form # Pasar el formulario con datos (y errores si los hay)

        if form.is_valid():
            data = form.cleaned_data
            method = data['method']
            tol = data['tolerance']
            niter = data['n_iterations']
            x0 = data.get('x0_xi')
            x1 = data.get('xs_x1')
            f_str = data.get('function_f')
            g_str = data.get('function_g')

            results = []
            message = ""

            try:
                # Procesar f(x) y sus derivadas
                f, f_prime, f_double_prime, error_msg = None, None, None, None
                if f_str:
                    f, f_prime, f_double_prime, error_msg = get_derivatives(f_str)
                    if error_msg: raise ValueError(error_msg)

                # Procesar g(x)
                g = None
                if g_str:
                    g, error_msg = parse_function(g_str)
                    if error_msg: raise ValueError(error_msg)

                # Llamar al método correspondiente
                if method == 'biseccion':
                    results, message = chapter1_methods.biseccion(f, x0, x1, tol, niter)
                elif method == 'regla_falsa':
                    results, message = chapter1_methods.regla_falsa(f, x0, x1, tol, niter)
                elif method == 'punto_fijo':
                    if not g: raise ValueError("Función g(x) no definida para Punto Fijo.")
                    results, message = chapter1_methods.punto_fijo(g, x0, tol, niter)
                elif method == 'newton':
                    results, message = chapter1_methods.newton(f, f_prime, x0, tol, niter)
                elif method == 'secante':
                    results, message = chapter1_methods.secante(f, x0, x1, tol, niter)
                elif method == 'newton_modificado':
                    results, message = chapter1_methods.newton_modificado(f, f_prime, f_double_prime, x0, tol, niter)

                # Renombrar 'f(xm)' a 'f_xm' para la plantilla (mejor nombre)
                # Y manejar NaNs para la plantilla
                formatted_results = []

                for row in results:
                    new_row = row.copy()
                    new_row['f_xm'] = new_row.pop('f(xm)', np.nan)  # Renombrar clave si existe

                    # Renombrar otras claves si vienen con nombres distintos
                    if 'Error' not in new_row and 'E' in new_row:
                        new_row['Error'] = new_row.pop('E')

                    # Formatear cada campo
                    for k, v in new_row.items():
                        if isinstance(v, float):
                            if np.isnan(v):
                                new_row[k] = '---'
                            elif k in ('f_xm', 'Error'):  # Mostrar en notación científica
                                new_row[k] = "{:.3e}".format(v)
                            else:  # Mostrar con 10 decimales para xi, xs, xm
                                new_row[k] = "{:.10f}".format(v)
                        elif v is None:
                            new_row[k] = '---'

                    formatted_results.append(new_row)

                context['results'] = formatted_results



            except ValueError as e:
                message = f"Error: {e}"
            except Exception as e:
                 message = f"Ocurrió un error inesperado: {e}"

            context['message'] = message

    return render(request, 'chapterOne/chapterOne.html', context)