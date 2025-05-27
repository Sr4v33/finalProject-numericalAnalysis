from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import get_template
from .forms import NumericalMethodForm
from . import methods as chapter1_methods
from core.utils.functions import parse_function, get_derivatives
import numpy as np
from xhtml2pdf import pisa 
import io 
import json


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
                                new_row[k] = "{:.11e}".format(v)
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




def run_all_methods(data):
    """
    Función auxiliar para ejecutar todos los métodos y recolectar datos.
    """
    tol = data['tolerance']
    niter = data['n_iterations']
    x0 = data.get('x0_xi')
    x1 = data.get('xs_x1')
    f_str = data.get('function_f')
    g_str = data.get('function_g') # <-- Obtener g(x)

    all_results = {}
    
    f, f_prime, f_double_prime, f_error_msg = get_derivatives(f_str)
    if f_error_msg:
        return {'error': f"Error con f(x): {f_error_msg}"}

    # --- Parsear g(x) ---
    g, g_error_msg = None, None
    if g_str:
        g, g_error_msg = parse_function(g_str) # Usamos parse_function, no get_derivatives
        if g_error_msg:
             all_results['Punto Fijo'] = {'results': [], 'message': f"Error con g(x): {g_error_msg}", 'iterations': 0, 'success': False, 'error': np.inf, 'root': np.nan}
    # --- Fin Parsear g(x) ---

    methods_to_run = {
        'Bisección': (chapter1_methods.biseccion, (f, x0, x1, tol, niter)),
        'Regla Falsa': (chapter1_methods.regla_falsa, (f, x0, x1, tol, niter)),
        'Newton': (chapter1_methods.newton, (f, f_prime, x0, tol, niter)),
        'Secante': (chapter1_methods.secante, (f, x0, x1, tol, niter)),
        'Newton Modificado': (chapter1_methods.newton_modificado, (f, f_prime, f_double_prime, x0, tol, niter)),
    }
    
    # --- Añadir Punto Fijo si es posible ---
    if g:
        methods_to_run['Punto Fijo'] = (chapter1_methods.punto_fijo, (g, x0, tol, niter))
    elif 'Punto Fijo' not in all_results: # Si no hubo error pero g no está, añadir mensaje
         all_results['Punto Fijo'] = {'results': [], 'message': "g(x) no proporcionado.", 'iterations': 0, 'success': False, 'error': np.inf, 'root': np.nan}
    # --- Fin Añadir Punto Fijo ---

    for name, (method_func, args) in methods_to_run.items():
        # Saltar si ya procesamos un error para este método (ej. g(x))
        if name in all_results:
            continue

        try:
            # Validar que los args necesarios no sean None
            if any(arg is None for arg in args if not callable(arg) and arg != tol and arg != niter):
                 raw_results, message = [], f"Parámetros incompletos ({name})."
            else:
                raw_results, message = method_func(*args)
            
            results = []
            for row in raw_results:
                new_row = row.copy()
                f_val = new_row.pop('f(xm)', new_row.get('f_xm', np.nan))
                new_row['f_xm'] = f_val
                results.append(new_row)

            iterations = len(results) -1 if results else 0
            is_success = 'raíz' in message or 'aprox.' in message or 'punto fijo' in message
            final_error = results[-1]['Error'] if results and 'Error' in results[-1] and not np.isnan(results[-1]['Error']) else np.inf
            root = results[-1]['xi'] if results else np.nan

            all_results[name] = {
                'results': results,
                'message': message,
                'iterations': iterations,
                'success': is_success,
                'error': final_error,
                'root': root,
            }
        except Exception as e:
            all_results[name] = {
                'results': [],
                'message': f"Error al ejecutar: {e}",
                'iterations': 0,
                'success': False,
                'error': np.inf,
                'root': np.nan,
            }

    return all_results

# ... (Las otras vistas: find_best_method, compare_methods_view, etc. se mantienen igual) ...

def find_best_method(all_results):

    best_method = None
    min_iterations = float('inf')

    for name, data in all_results.items():
        if data['success'] and data['iterations'] < min_iterations:
            min_iterations = data['iterations']
            best_method = name

    return best_method, min_iterations if best_method else None

def compare_methods_view(request):

    form = NumericalMethodForm(request.POST or None)
    context = {'form': form}

    if request.method == 'POST' and form.is_valid():
        data = form.cleaned_data
        all_results = run_all_methods(data)
        
        if 'error' in all_results:
            context['global_error'] = all_results['error']
        else:
            best_method, min_iter = find_best_method(all_results)
            context['all_results'] = all_results
            context['best_method'] = best_method
            context['min_iter'] = min_iter
            context['input_data'] = data # Guardar datos de entrada

            # --- Guardar en sesión para el PDF ---
            request.session['report_data'] = {
                'all_results': all_results,
                'best_method': best_method,
                'min_iter': min_iter,
                'input_data': {k: str(v) for k, v in data.items()}, # Convertir a str
            }

    return render(request, 'chapterOne/compare_report.html', context)



def render_to_pdf(template_src, context_dict={}):

    template = get_template(template_src)
    html = template.render(context_dict)
    result = io.BytesIO() # Buffer en memoria
    
    # Crear el PDF
    pdf = pisa.pisaDocument(io.BytesIO(html.encode("UTF-8")), result)
    
    if not pdf.err:
        return HttpResponse(result.getvalue(), content_type='application/pdf')
    return HttpResponse(f"Error al generar PDF: {pdf.err}")



def download_pdf_report_view(request):

    report_data = request.session.get('report_data')

    if not report_data:
        return HttpResponse("No hay datos de informe para generar el PDF. Por favor, primero genera una comparación.", status=404)

    # Añadir información para el PDF
    report_data['is_pdf'] = True 
    
    return render_to_pdf('chapterOne/pdf_template.html', report_data)






def graph_function_view(request):
    function_string = request.GET.get('function', None)
    context = {'function_string': function_string}

    if function_string:
        func, error_msg = parse_function(function_string)

        if error_msg:
            context['error'] = f"Error al parsear la función: {error_msg}"
        else:
            try:
                # Usar paso de 0.1
                x_values = np.arange(-100, 100.1, 0.1).tolist()
                x_for_eval = np.array(x_values)
                y_values = []

                with np.errstate(divide='ignore', invalid='ignore'):
                    y_raw = func(x_for_eval)

                for y in y_raw:
                    if np.isfinite(y):
                        y_values.append(y)
                    else:
                        y_values.append(None)

                context['plot_data'] = json.dumps({
                    'x': x_values,
                    'y': y_values
                })

            except Exception as e:
                context['error'] = f"Error al generar datos para el gráfico: {e}"

    return render(request, 'chapterOne/graph.html', context)

