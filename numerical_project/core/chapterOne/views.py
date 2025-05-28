from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import get_template
from .forms import NumericalMethodForm
from . import methods as chapter1_methods
from core.utils.functions import parse_function, get_derivatives
import numpy as np
import sympy
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
                    results, message = chapter1_methods.punto_fijo(f, g, x0, tol, niter)
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
                    # --- INICIO DEL BLOQUE INDENTADO ---
                    new_row = row.copy()
                    new_row['f_xm'] = new_row.pop('f(xn)', new_row.pop('f(xm)', np.nan)) # Ajustado para buscar f(xn) primero
                    
                    # Renombrar 'Error' si es necesario
                    if 'Error' not in new_row and 'E' in new_row:
                        new_row['Error'] = new_row.pop('E')
                    
                    # Asegurarse que 'Error' exista si es necesario (ej. si usas .pop('Error'))
                    if 'Error' not in new_row:
                        new_row['Error'] = np.nan # o None, o lo que corresponda

                    # Formatear cada campo
                    for k, v in new_row.items():
                        if isinstance(v, float):
                            if np.isnan(v):
                                new_row[k] = '---'
                            # Ajusta las claves según las que devuelven tus métodos (f_xm, Error, etc.)
                            elif k in ('f_xm', 'Error', 'f(xn)'):  # Mostrar en notación científica
                                new_row[k] = "{:.3e}".format(v)
                            else:  # Mostrar con 10 decimales (usar 10f, no 11e)
                                new_row[k] = "{:.10f}".format(v) # Corregido: .10f o .11f si prefieres
                        elif v is None:
                            new_row[k] = '---'

                    formatted_results.append(new_row) # <--- ¡ASEGÚRATE DE QUE ESTÁ DENTRO DEL BUCLE!
                    # --- FIN DEL BLOQUE INDENTADO ---
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
        methods_to_run['Punto Fijo'] = (chapter1_methods.punto_fijo, (f, g, x0, tol, niter))
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



def find_best_method(all_results):
    best_method_name = None
    min_iterations = float('inf')
    min_error_at_min_iterations = float('inf')

    for name, data in all_results.items():
        # Asegurarse de que 'iterations' y 'error' existan y sean válidos
        if not data.get('success'): # Solo considerar métodos exitosos
            continue

        current_iterations = data.get('iterations', float('inf'))
        current_error = data.get('error', float('inf'))

        # Si el error es NaN, trátalo como infinito para que no sea elegido como el mejor
        if np.isnan(current_error):
            current_error = float('inf')

        # Criterio 1: Menor número de iteraciones
        if current_iterations < min_iterations:
            min_iterations = current_iterations
            min_error_at_min_iterations = current_error
            best_method_name = name
        elif current_iterations == min_iterations:
            # Criterio 2: Si las iteraciones son iguales, desempatar por menor error
            if current_error < min_error_at_min_iterations:
                min_error_at_min_iterations = current_error
                best_method_name = name
                # No necesitamos actualizar min_iterations aquí porque ya es el mínimo
    
    # Devolver el nombre del mejor método y sus iteraciones (o None si ninguno tuvo éxito)
    if best_method_name:
        return best_method_name, min_iterations
    else:
        return None, None

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
                
                # --- MODIFICACIÓN CLAVE PARA PRUEBAS ---
                if "exp" in function_string.lower() and ("(" in function_string and ")" in function_string): # Heurística simple
                    print(f"Función exponencial detectada: '{function_string}'. Usando rango reducido para graficar.")
                    x_values_np = np.arange(-50, 50.1, 0.1) # Rango mucho más pequeño
                elif "log" in function_string.lower():
                     print(f"Función logarítmica detectada: '{function_string}'. Usando rango positivo.")
                     x_values_np = np.arange(-50, 50.1, 0.1) # Rango positivo y acotado
                else:
                    # Rango general, quizás también más acotado que -100 a 100
                    print(f"Función general: '{function_string}'. Usando rango -20 a 20.")
                    x_values_np = np.arange(-100, 100.1, 0.1)


                x_values = x_values_np.tolist()   
                y_values = []
                
                
                if func is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_raw = func(x_values_np) # Usa el array numpy aquí

                    for y_val_single in y_raw: # Itera sobre los elementos del array y_raw
                        if np.isfinite(y_val_single):
                            y_values.append(y_val_single)
                        else:
                            y_values.append(None) # Python None se convierte a null en JSON

                    context['plot_data'] = json.dumps({
                        'x': x_values,
                        'y': y_values
                    })
                    # Imprime para depurar qué se envía al template
                    # print("Plot Data para el template:", context['plot_data'][:200] + "...")
                else:
                    context['error'] = "No se pudo interpretar la función proporcionada."


            except Exception as e:
                context['error'] = f"Error al generar datos para el gráfico: {e}"
                print(f"Error en graph_function_view al generar datos: {e}") # Imprime error en consola del servidor

    return render(request, 'chapterOne/graph.html', context)
