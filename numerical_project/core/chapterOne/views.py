from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import get_template
from .forms import NumericalMethodForm
from .import methods as chapter1_methods
from core.utils.functions import parse_function, get_derivatives
import numpy as np
from xhtml2pdf import pisa
import io
import json

# ==============================================================================
# --- Vistas de Métodos Numéricos (Capítulo 1) ---
# ==============================================================================

def chapter_one_view(request):
    """
    Gestiona la vista principal para ejecutar un método numérico individual.

    Maneja solicitudes GET para mostrar el formulario y solicitudes POST para
    procesar los datos del formulario, ejecutar el método seleccionado y
    mostrar los resultados o errores.

    Args:
        request: Objeto HttpRequest de Django.

    Returns:
        HttpResponse: Renderiza la plantilla 'chapterOne/chapterOne.html'
                      con el contexto apropiado.
    """
    form = NumericalMethodForm()
    context = {'form': form}

    # Si la solicitud es POST, procesar el formulario
    if request.method == 'POST':
        form = NumericalMethodForm(request.POST)
        context['form'] = form  # Actualizar contexto con datos (y posibles errores)

        if form.is_valid():
            # Obtener datos limpios del formulario
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
                # --- Procesamiento de Funciones ---
                f, f_prime, f_double_prime, error_msg = None, None, None, None
                if f_str:
                    # Obtener f(x) y sus derivadas
                    f, f_prime, f_double_prime, error_msg = get_derivatives(f_str)
                    if error_msg: raise ValueError(error_msg)

                g = None
                if g_str:
                    # Obtener g(x) (para Punto Fijo)
                    g, error_msg = parse_function(g_str)
                    if error_msg: raise ValueError(error_msg)

                # --- Ejecución del Método Seleccionado ---
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

                # --- Formateo de Resultados para la Plantilla ---
                formatted_results = []
                for row in results:
                    new_row = row.copy()
                    # Unificar 'f(xm)' o 'f(xn)' a 'f_xm'
                    new_row['f_xm'] = new_row.pop('f(xn)', new_row.pop('f(xm)', np.nan))

                    # Unificar 'Error' o 'E'
                    if 'Error' not in new_row and 'E' in new_row:
                        new_row['Error'] = new_row.pop('E')
                    elif 'Error' not in new_row:
                        new_row['Error'] = np.nan # Asegurar que 'Error' exista

                    # Formatear números para visualización
                    for k, v in new_row.items():
                        if isinstance(v, float):
                            if np.isnan(v):
                                new_row[k] = '---'  # Mostrar '---' para NaN
                            # Formato científico para f(x) y Error
                            elif k in ('f_xm', 'Error'):
                                new_row[k] = "{:.3e}".format(v)
                            # Formato decimal para valores de x
                            else:
                                new_row[k] = "{:.10f}".format(v)
                        elif v is None:
                            new_row[k] = '---'  # Mostrar '---' para None

                    formatted_results.append(new_row)

                context['results'] = formatted_results

            except ValueError as e:
                # Capturar errores de validación o parseo
                message = f"Error de Validación: {e}"
            except Exception as e:
                # Capturar otros errores inesperados
                message = f"Ocurrió un error inesperado: {e}"

            context['message'] = message

    # Renderizar la plantilla con el contexto
    return render(request, 'chapterOne/chapterOne.html', context)


def run_all_methods(data):
    """
    Ejecuta todos los métodos numéricos disponibles con los datos proporcionados
    y recopila sus resultados para comparación.

    Args:
        data (dict): Un diccionario con los datos del formulario
                     (tolerancia, iteraciones, x0, x1, f(x), g(x)).

    Returns:
        dict: Un diccionario donde cada clave es el nombre de un método y
              su valor es otro diccionario con 'results', 'message',
              'iterations', 'success', 'error' y 'root'.
              O un diccionario con {'error': 'mensaje'} si hay un error global.
    """
    # Extraer datos de entrada
    tol = data['tolerance']
    niter = data['n_iterations']
    x0 = data.get('x0_xi')
    x1 = data.get('xs_x1')
    f_str = data.get('function_f')
    g_str = data.get('function_g')

    all_results = {}

    # Parsear f(x) y sus derivadas
    f, f_prime, f_double_prime, f_error_msg = get_derivatives(f_str)
    if f_error_msg:
        return {'error': f"Error con f(x): {f_error_msg}"}

    # Parsear g(x)
    g, g_error_msg = None, None
    if g_str:
        g, g_error_msg = parse_function(g_str)
        if g_error_msg:
            # Guardar error específico para Punto Fijo si g(x) falla
            all_results['Punto Fijo'] = {'results': [], 'message': f"Error con g(x): {g_error_msg}", 'iterations': 0, 'success': False, 'error': np.inf, 'root': np.nan}

    # Definir los métodos a ejecutar y sus argumentos
    methods_to_run = {
        'Bisección': (chapter1_methods.biseccion, (f, x0, x1, tol, niter)),
        'Regla Falsa': (chapter1_methods.regla_falsa, (f, x0, x1, tol, niter)),
        'Newton': (chapter1_methods.newton, (f, f_prime, x0, tol, niter)),
        'Secante': (chapter1_methods.secante, (f, x0, x1, tol, niter)),
        'Newton Modificado': (chapter1_methods.newton_modificado, (f, f_prime, f_double_prime, x0, tol, niter)),
    }

    # Añadir Punto Fijo solo si g(x) es válido
    if g:
        methods_to_run['Punto Fijo'] = (chapter1_methods.punto_fijo, (f, g, x0, tol, niter))
    elif 'Punto Fijo' not in all_results: # Si no hubo error pero g no existe
        all_results['Punto Fijo'] = {'results': [], 'message': "g(x) no proporcionado.", 'iterations': 0, 'success': False, 'error': np.inf, 'root': np.nan}

    # Iterar y ejecutar cada método
    for name, (method_func, args) in methods_to_run.items():
        if name in all_results: # Saltar si ya tiene un resultado (ej. error de g(x))
            continue

        try:
            # Validar que los argumentos numéricos necesarios no sean None
            # (Se excluyen funciones, tol y niter de esta verificación)
            required_args = [arg for arg in args if not callable(arg) and not isinstance(arg, (int, float))]
            if any(arg is None for arg in required_args):
                 raw_results, message = [], f"Parámetros incompletos ({name})."
            else:
                 raw_results, message = method_func(*args)

            # Procesar y estandarizar resultados
            results = []
            for row in raw_results:
                new_row = row.copy()
                new_row['f_xm'] = new_row.pop('f(xm)', new_row.get('f_xn', np.nan))
                results.append(new_row)

            # Extraer métricas clave
            iterations = len(results) -1 if results else 0
            is_success = 'raíz' in message or 'aprox.' in message or 'punto fijo' in message
            final_error = results[-1].get('Error', np.inf) if results else np.inf
            # Tratar NaN en error como infinito para la comparación
            final_error = np.inf if np.isnan(final_error) else final_error
            root = results[-1].get('xi', np.nan) if results else np.nan

            # Guardar resultados del método
            all_results[name] = {
                'results': results, 'message': message, 'iterations': iterations,
                'success': is_success, 'error': final_error, 'root': root,
            }
        except Exception as e:
            # Capturar errores durante la ejecución del método
            all_results[name] = {
                'results': [], 'message': f"Error al ejecutar: {e}", 'iterations': 0,
                'success': False, 'error': np.inf, 'root': np.nan,
            }

    return all_results


def find_best_method(all_results):
    """
    Determina el "mejor" método numérico basado en los resultados obtenidos.

    El criterio es:
    1. Debe haber sido exitoso (haber encontrado una raíz).
    2. Menor número de iteraciones.
    3. En caso de empate en iteraciones, menor error final.

    Args:
        all_results (dict): El diccionario de resultados de run_all_methods.

    Returns:
        tuple: (nombre_mejor_metodo, iteraciones_minimas) o (None, None)
               si ningún método fue exitoso.
    """
    best_method_name = None
    min_iterations = float('inf')
    min_error_at_min_iterations = float('inf')

    for name, data in all_results.items():
        # Solo considerar métodos que encontraron una raíz
        if not data.get('success'):
            continue

        current_iterations = data.get('iterations', float('inf'))
        current_error = data.get('error', float('inf'))

        # Tratar NaN como infinito para la comparación
        current_error = float('inf') if np.isnan(current_error) else current_error

        # Aplicar criterios de selección
        if current_iterations < min_iterations:
            min_iterations = current_iterations
            min_error_at_min_iterations = current_error
            best_method_name = name
        elif current_iterations == min_iterations:
            if current_error < min_error_at_min_iterations:
                min_error_at_min_iterations = current_error
                best_method_name = name

    # Devolver el mejor encontrado o None si no hubo éxitos
    return (best_method_name, min_iterations) if best_method_name else (None, None)


def compare_methods_view(request):
    """
    Gestiona la vista para comparar todos los métodos numéricos.

    Maneja GET para mostrar el formulario y POST para procesarlo,
    ejecutar todos los métodos, encontrar el mejor, guardar los datos en
    sesión para el PDF y renderizar la plantilla de comparación.

    Args:
        request: Objeto HttpRequest de Django.

    Returns:
        HttpResponse: Renderiza la plantilla 'chapterOne/compare_report.html'.
    """
    form = NumericalMethodForm(request.POST or None) # Acepta POST o inicializa vacío
    context = {'form': form}

    if request.method == 'POST' and form.is_valid():
        data = form.cleaned_data
        all_results = run_all_methods(data) # Ejecutar todos los métodos

        if 'error' in all_results:
            # Mostrar error global si ocurrió (ej. f(x) inválida)
            context['global_error'] = all_results['error']
        else:
            # Encontrar el mejor método
            best_method, min_iter = find_best_method(all_results)
            context['all_results'] = all_results
            context['best_method'] = best_method
            context['min_iter'] = min_iter
            context['input_data'] = data # Guardar datos de entrada

            # --- Guardar en sesión para el PDF ---
            # Se convierte a string para asegurar compatibilidad con JSON (motor de sesión)
            request.session['report_data'] = {
                'all_results': all_results,
                'best_method': best_method,
                'min_iter': min_iter,
                'input_data': {k: str(v) for k, v in data.items()},
            }

    return render(request, 'chapterOne/compare_report.html', context)


def render_to_pdf(template_src, context_dict={}):
    """
    Renderiza una plantilla HTML a un objeto HttpResponse en formato PDF.

    Args:
        template_src (str): La ruta a la plantilla HTML.
        context_dict (dict): El diccionario de contexto para la plantilla.

    Returns:
        HttpResponse: Una respuesta HTTP con el PDF generado o un mensaje
                      de error.
    """
    template = get_template(template_src)
    html = template.render(context_dict) # Renderizar HTML
    result = io.BytesIO() # Crear un buffer en memoria

    # Convertir HTML a PDF usando xhtml2pdf (pisa)
    pdf = pisa.pisaDocument(io.BytesIO(html.encode("UTF-8")), result)

    # Si no hubo errores, devolver el PDF
    if not pdf.err:
        return HttpResponse(result.getvalue(), content_type='application/pdf')
    # Si hubo error, devolver un mensaje
    return HttpResponse(f"Error al generar PDF: {pdf.err}")


def download_pdf_report_view(request):
    """
    Gestiona la descarga del informe de comparación en formato PDF.

    Recupera los datos del informe guardados en la sesión y llama a
    'render_to_pdf' para generar y devolver el archivo PDF.

    Args:
        request: Objeto HttpRequest de Django.

    Returns:
        HttpResponse: La respuesta PDF o un mensaje si no hay datos.
    """
    # Obtener datos de la sesión
    report_data = request.session.get('report_data')

    # Verificar si hay datos
    if not report_data:
        return HttpResponse("No hay datos de informe para generar el PDF. Por favor, primero genera una comparación.", status=404)

    # Añadir una bandera para posible lógica específica del PDF en la plantilla
    report_data['is_pdf'] = True

    # Renderizar y devolver el PDF
    return render_to_pdf('chapterOne/pdf_template.html', report_data)


def graph_function_view(request):
    """
    Gestiona la vista para mostrar el gráfico de una función.

    Recibe la función como una cadena de texto (GET), la parsea, genera
    puntos (x, y) para el gráfico, y renderiza una plantilla que
    mostrará el gráfico (probablemente usando JavaScript).

    Args:
        request: Objeto HttpRequest de Django.

    Returns:
        HttpResponse: Renderiza la plantilla 'chapterOne/graph.html'.
    """
    function_string = request.GET.get('function', None)
    context = {'function_string': function_string}

    if function_string:
        # Parsear la función
        func, error_msg = parse_function(function_string)

        if error_msg:
            context['error'] = f"Error al parsear la función: {error_msg}"
        else:
            try:
                # --- Selección de Rango para Graficar ---
                # Se usan heurísticas para ajustar el rango y evitar problemas
                # con funciones exponenciales o logarítmicas en rangos muy amplios.
                if "exp" in function_string.lower():
                    x_values_np = np.arange(-50, 50.1, 0.1)
                elif "log" in function_string.lower():
                    x_values_np = np.arange(0.1, 50.1, 0.1) # Empezar > 0 para log
                else:
                    x_values_np = np.arange(-100, 100.1, 0.1) # Rango por defecto

                x_values = x_values_np.tolist()
                y_values = []

                if func is not None:
                    # Calcular Y, ignorando errores de división/inválidos (ej. log(0), 1/0)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_raw = func(x_values_np)

                    # Filtrar infinitos y NaN, reemplazándolos con None para JSON
                    for y_val_single in y_raw:
                        if np.isfinite(y_val_single):
                            y_values.append(y_val_single)
                        else:
                            y_values.append(None) # None se traduce a 'null' en JSON

                    # Convertir datos a JSON para pasarlos a JavaScript
                    context['plot_data'] = json.dumps({
                        'x': x_values,
                        'y': y_values
                    })
                else:
                    context['error'] = "No se pudo interpretar la función proporcionada."

            except Exception as e:
                context['error'] = f"Error al generar datos para el gráfico: {e}"
                print(f"Error en graph_function_view: {e}") # Log del error en servidor

    # Renderizar la plantilla del gráfico
    return render(request, 'chapterOne/graph.html', context)