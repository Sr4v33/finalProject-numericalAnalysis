# chapterThree/views.py

# Importaciones de Django y Python estándar
from django.shortcuts import render  # Para renderizar plantillas HTML.
from django.http import HttpRequest, HttpResponse, JsonResponse # Clases para manejar peticiones y respuestas HTTP.
from django.template.loader import get_template # Para cargar plantillas HTML como objetos.

# Importaciones locales (de nuestra aplicación)
from .forms import InterpolationForm, PointFormSet # Nuestros formularios definidos en forms.py.
from .methods import ( # Nuestras funciones de cálculo definidas en methods.py.
    calculate_vandermonde, calculate_newton, calculate_lagrange,
    calculate_linear_spline, calculate_cubic_spline, evaluate_interpolation
)

# Importaciones de bibliotecas de terceros
from xhtml2pdf import pisa  # Para convertir HTML a PDF.
import numpy as np          # Para cálculos numéricos (usado en errores y cálculos).
import json                 # Para manejar datos en formato JSON (para gráficos).
# import matplotlib.pyplot as plt # Comentado, parece no usarse directamente aquí para generar imagen.
import io                   # Para manejar flujos de bytes en memoria (útil para PDF).


# --- VISTAS ---

def interpolation_view(request: HttpRequest) -> HttpResponse:
    """
    Vista principal para la interpolación. Maneja tanto la visualización
    inicial del formulario como el procesamiento de los datos enviados (POST).
    """
    # Contexto inicial: datos que se pasarán a la plantilla HTML.
    context = {
        'page_title': "Capítulo 3: Interpolación", # Título de la página.
        'form': InterpolationForm(), # Un formulario principal vacío.
        'formset': PointFormSet(initial=[{'x': '', 'y': ''}, {'x': '', 'y': ''}]), # Un formset con 2 filas iniciales vacías.
        'results': None, # Inicialmente no hay resultados.
        'error_message': None, # Inicialmente no hay errores.
    }

    # Si la petición es de tipo POST, significa que el usuario envió el formulario.
    if request.method == "POST":
        # Crea instancias del formulario y formset con los datos enviados (request.POST).
        form = InterpolationForm(request.POST)
        formset = PointFormSet(request.POST)
        # Actualiza el contexto con los formularios (para mostrarlos de nuevo si hay errores o resultados).
        context['form'] = form
        context['formset'] = formset

        # Verifica si tanto el formulario principal como el formset son válidos.
        # Esto ejecuta las validaciones definidas en forms.py (incluido BasePointFormSet.clean).
        if form.is_valid() and formset.is_valid():
            # Obtiene el método seleccionado y los puntos ingresados (filtrando los eliminados).
            method = form.cleaned_data['method']
            points_data = [
                data for data in formset.cleaned_data if data and not data.get('DELETE')
            ]
            
            # Obtiene el nombre legible del método para mostrarlo en los resultados.
            context['method_name'] = dict(form.fields['method'].choices).get(method, method)

            # Intenta realizar el cálculo. Usa un bloque try...except para capturar errores.
            try:
                # Llama a la función de cálculo correspondiente según el método.
                if method == 'vandermonde':
                    # Desempaqueta todos los resultados devueltos por la función.
                    matrix_a, vector_b, coeffs, poly_str_normal, plot_data, poly_sympy, poly_str_latex = calculate_vandermonde(points_data)
                    # Construye el diccionario de resultados para la plantilla.
                    context['results'] = {
                        'type': 'vandermonde', # Identifica el tipo para la plantilla.
                        'points': points_data,
                        'matrixA': matrix_a,
                        'B': vector_b,
                        'coeffs': [f"{c:.6f}" for c in coeffs], # Formatea coeficientes.
                        'polynom': poly_str_normal,
                        'plot_data': plot_data # Datos para el gráfico.
                    }
                
                elif method == 'newton':
                    headers, display_table, newton_coeffs_divided_differences, poly_str_expanded, poly_str_newton_native, plot_data, poly_sympy, poly_str_latex = calculate_newton(points_data)
                    context['results'] = {
                        'type': 'newton',
                        'points': points_data,
                        'headers': headers,
                        'table': display_table, # Tabla de diferencias.
                        'coeffs': newton_coeffs_divided_differences,
                        'polynom': poly_str_expanded, # Forma expandida.
                        'polynom_newton_form': poly_str_newton_native, # Forma nativa.
                        'plot_data': plot_data
                    }
                
                elif method == 'lagrange':
                    lagrange_basis_for_table, poly_str_lagrange_native, poly_str_expanded, plot_data, poly_sympy, poly_str_latex = calculate_lagrange(points_data)
                    context['results'] = {
                        'type': 'lagrange',
                        'points': points_data,
                        'lagrange_basis_polynomials': lagrange_basis_for_table,
                        'polynom_lagrange_form': poly_str_lagrange_native,
                        'polynom': poly_str_expanded,
                        'plot_data': plot_data
                    }

                elif method == 'spline_linear':
                    coeffs_table_list, tracers_list, plot_data, spline_sympy_list, poly_str_latex = calculate_linear_spline(points_data)
                    # Formatea los tramos para mostrarlos.
                    polynom_parts = [f"S_{{{item['i']}}}(x) = {item['tracer_str']}" for item in tracers_list]
                    polynom_string = "; \\; ".join(polynom_parts) # Une con separador LaTeX.
                    context['results'] = {
                        'type': 'spline_linear',
                        'points': points_data,
                        'coeffs_table': coeffs_table_list,
                        'tracers_list': tracers_list, # Lista de tramos.
                        'plot_data': plot_data,
                        'polynom': polynom_string # Cadena con todos los tramos.
                    }
                
                elif method == 'spline_cubic':
                    coeffs_table_list, tracers_list, plot_data, spline_sympy_list, poly_str_latex = calculate_cubic_spline(points_data)
                    # Formatea los tramos para mostrarlos (usa LaTeX directamente).
                    polynom_parts = [f"S_{{{item['i']}}}(x) = {item['tracer_str']}" for item in tracers_list]
                    polynom_string = "; \\; ".join(polynom_parts)
                    context['results'] = {
                        'type': 'spline_cubic',
                        'points': points_data,
                        'coeffs_table': coeffs_table_list,
                        'tracers_list': tracers_list,
                        'plot_data': plot_data,
                        'polynom': polynom_string
                    }

                else:
                    context['error_message'] = f"El método '{method}' aún no está implementado."
                
                # Si hay resultados y el método no es un spline, añade el LaTeX y el objeto Sympy al contexto.
                if context.get('results') and 'poly_sympy' in locals():
                    context['results']['poly_sympy_obj'] = poly_sympy # Objeto para evaluar.
                    context['results']['poly_latex'] = poly_str_latex # String para mostrar.
                # Si es un spline, añade la lista de objetos Sympy y el string LaTeX.
                elif context.get('results') and 'spline_sympy_list' in locals():
                    context['results']['poly_sympy_obj'] = spline_sympy_list
                    context['results']['poly_latex'] = poly_str_latex

            # Captura errores específicos de cálculo (ej: matriz singular) o de valor (ej: x repetidos).
            except (np.linalg.LinAlgError, ValueError) as e:
                context['error_message'] = f"Error al calcular: {e}"
            # Captura cualquier otro error inesperado.
            except Exception as e:
                context['error_message'] = f"Ocurrió un error inesperado: {e}"

    # Renderiza la plantilla 'chapterThree.html' con el contexto (ya sea inicial o con resultados/errores).
    return render(request, 'chapterThree/chapterThree.html', context)


def graph_view(request: HttpRequest) -> HttpResponse:
    """
    Vista dedicada a renderizar la página del gráfico.
    Recibe los datos para graficar a través de parámetros GET.
    Prioriza el uso de 'plot_data' si está disponible.
    """
    try:
        # Intenta obtener la cadena JSON 'plot_data' de los parámetros GET.
        plot_data_str = request.GET.get('plot_data', None)
        # Obtiene el nombre de la función (título) para el gráfico.
        function_string = request.GET.get('function_name', 'Interpolación')

        # Si se recibió 'plot_data', se usa directamente.
        if plot_data_str:
            try:
                # Intenta cargar el JSON para validar que sea correcto (opcional).
                json.loads(plot_data_str)
                plot_data_to_render = plot_data_str # Usa la cadena tal cual.
            except json.JSONDecodeError:
                raise ValueError("El 'plot_data' recibido no es un JSON válido.")

        # Si NO se recibió 'plot_data', intenta construirlo (Fallback, menos preferido).
        else:
            # Obtiene los puntos originales X e Y.
            original_x = json.loads(request.GET.get('original_x', '[]'))
            original_y = json.loads(request.GET.get('original_y', '[]'))

            # Validaciones básicas para el fallback.
            if not original_x or not original_y:
                raise ValueError("Faltan datos para la interpolación (fallback).")
            if len(original_x) != len(original_y):
                raise ValueError("Longitud de X y Y no coincide (fallback).")

            # Convierte a arrays de Numpy.
            x = np.array(original_x)
            y = np.array(original_y)

            # Ajusta un polinomio usando polyfit (esto puede diferir del método original).
            coeffs = np.polyfit(x, y, deg=len(x) - 1)
            poly = np.poly1d(coeffs)

            # Genera puntos interpolados para una curva suave (ampliando un poco el rango).
            min_x, max_x = min(x), max(x)
            interpolated_x = np.linspace(min_x - 50, max_x + 50, 1000).tolist()
            interpolated_y = poly(interpolated_x).tolist()

            # Crea la cadena JSON 'plot_data'.
            plot_data_to_render = json.dumps({
                'original_x': original_x,
                'original_y': original_y,
                'interpolated_x': interpolated_x,
                'interpolated_y': interpolated_y,
            })

        # Prepara el contexto para la plantilla del gráfico.
        context = {
            'function_string': function_string,
            'plot_data': plot_data_to_render,
            'error': None,
        }
        # Renderiza la plantilla del gráfico.
        return render(request, 'chapterThree/graph_ch3.html', context)

    # Maneja errores de JSON o de valor.
    except (json.JSONDecodeError, ValueError) as e:
        context = {'error': f"Error al procesar los datos para el gráfico: {e}"}
        return render(request, 'chapterThree/graph_ch3.html', context)
    # Maneja cualquier otro error.
    except Exception as e:
        context = {'error': f"Ocurrió un error inesperado: {e}"}
        return render(request, 'chapterThree/graph_ch3.html', context)

def compare_methods_view(request: HttpRequest) -> HttpResponse:
    """
    Vista para comparar todos los métodos de interpolación.
    Calcula cada método, evalúa en un punto dado y muestra los errores.
    Guarda los resultados en la sesión para poder generar un PDF.
    """
    # Contexto inicial.
    context = {
        'page_title': "Reporte de Comparación de Métodos de Interpolación",
        'form': InterpolationForm(),
        'formset': PointFormSet(),
    }

    # Esta vista solo debe ser accesible vía POST (desde el formulario).
    if request.method != "POST":
        context['error_message'] = "Acceso inválido a la comparación. Por favor, usa el formulario."
        return render(request, 'chapterThree/chapterThree.html', context)

    # Carga los formularios con los datos POST.
    form = InterpolationForm(request.POST)
    formset = PointFormSet(request.POST)

    # Valida los formularios.
    if form.is_valid() and formset.is_valid():
        # Extrae los puntos y los valores X e Y para evaluar el error.
        points_data = [
            data for data in formset.cleaned_data if data and not data.get('DELETE')
        ]
        x_eval = form.cleaned_data.get('x_eval')
        y_eval = form.cleaned_data.get('y_eval')

        # Si no se ingresaron los puntos para evaluar el error, regresa a la vista principal con un mensaje.
        if x_eval is None or y_eval is None:
            error_context = {
                'page_title': "Capítulo 3: Interpolación",
                'form': form,
                'formset': formset,
                'error_message': "Para comparar métodos, debes ingresar un valor para 'X para Evaluar Error' y 'Y Real en X_eval'.",
            }
            return render(request, 'chapterThree/chapterThree.html', error_context)

        # Diccionario para almacenar los datos del reporte (se usará en HTML y PDF).
        report_content_data = {
            'x_eval_report': x_eval,
            'y_eval_report': y_eval,
            'points_data_report': points_data,
            'comparison_results': [], # Se llenará más adelante.
            'best_method': None,     # Se llenará más adelante.
            'page_title': "Reporte de Comparación de Métodos de Interpolación"
        }

        # Diccionario que mapea nombres de métodos a sus funciones de cálculo.
        methods_to_run = {
            "Vandermonde": calculate_vandermonde,
            "Newton": calculate_newton,
            "Lagrange": calculate_lagrange,
            "Spline Lineal": calculate_linear_spline,
            "Spline Cúbico": calculate_cubic_spline,
        }

        comparison_results_list = []

        # Itera sobre cada método para calcularlo y evaluarlo.
        for method_name, calculate_function in methods_to_run.items():
            poly_data_for_eval = None # Objeto Sympy o lista para evaluar.
            representative_poly_str = "N/A" # String LaTeX para mostrar.
            
            try:
                # Llama a la función de cálculo y desempaqueta SOLO lo necesario:
                # el objeto Sympy/lista (poly_data_for_eval) y el string LaTeX.
                # Nota: El desempaquetado debe coincidir con lo que devuelve cada función.
                if method_name == "Vandermonde":
                    _, _, _, _, _, poly_data_for_eval, representative_poly_str = calculate_function(points_data)
                elif method_name == "Newton":
                    _, _, _, _, _, _, poly_data_for_eval, representative_poly_str = calculate_function(points_data)
                elif method_name == "Lagrange":
                    _, _, _, _, poly_data_for_eval, representative_poly_str = calculate_function(points_data)
                elif method_name in ["Spline Lineal", "Spline Cúbico"]:
                    _, _, _, poly_data_for_eval, representative_poly_str = calculate_function(points_data)
                
                # Evalúa el polinomio/spline en el punto x_eval.
                y_predicted = evaluate_interpolation(poly_data_for_eval, x_eval)
                
                # Calcula el error absoluto. Maneja el caso de NaN (Not a Number).
                if np.isnan(y_predicted):
                    error = float('inf') # Error infinito si no se pudo evaluar.
                    y_predicted_display = "Error (NaN)"
                else:
                    error = abs(y_eval - y_predicted)
                    y_predicted_display = f"{y_predicted:.6f}"

                # Añade los resultados de este método a la lista.
                comparison_results_list.append({
                    'name': method_name,
                    'poly_str': representative_poly_str,
                    'y_predicted': y_predicted_display,
                    'error': error
                })
            # Captura errores específicos durante el cálculo o evaluación de un método.
            except (ValueError, np.linalg.LinAlgError) as ve:
                print(f"Error calculando {method_name}: {ve}")
                comparison_results_list.append({
                    'name': method_name, 'poly_str': f"Error: {ve}",
                    'y_predicted': "Error", 'error': float('inf')
                })
            # Captura cualquier otro error para ese método.
            except Exception as e:
                print(f"Error general calculando {method_name}: {e}")
                comparison_results_list.append({
                    'name': method_name, 'poly_str': "Error en cálculo",
                    'y_predicted': "Error", 'error': float('inf')
                })
        
        # Almacena la lista de resultados en el diccionario del reporte.
        report_content_data['comparison_results'] = comparison_results_list
        
        # Encuentra el mejor método (menor error, excluyendo infinitos).
        valid_results = [res for res in comparison_results_list if isinstance(res['error'], (int, float)) and res['error'] != float('inf')]
        if valid_results:
            report_content_data['best_method'] = min(valid_results, key=lambda x: x['error'])

        # === GUARDAR DATOS PARA EL PDF EN LA SESIÓN ===
        # Se guardan los datos del reporte en la sesión del usuario.
        # Esto permite que la vista 'download_pdf_report_view' pueda acceder a ellos
        # cuando el usuario haga clic en el botón de descarga.
        request.session['report_data'] = report_content_data
        # ==============================================
        
        # Actualiza el contexto principal con los datos del reporte para mostrarlo en HTML.
        context.update(report_content_data)
        
        # Renderiza la plantilla del reporte de comparación.
        return render(request, 'chapterThree/compare_report_ch3.html', context)

    # Si los formularios no son válidos, regresa a la vista principal mostrando los errores.
    else:
        context['form'] = form
        context['formset'] = formset
        context['error_message'] = "Hubo errores en el formulario. Por favor, revisa los campos."
        return render(request, 'chapterThree/chapterThree.html', context)


# --- FUNCIONES AUXILIARES (PDF) ---

def render_to_pdf(template_src, context_dict={}):
    """
    Función auxiliar para renderizar una plantilla HTML y convertirla en un PDF.

    Args:
        template_src (str): La ruta a la plantilla HTML.
        context_dict (dict): El diccionario de contexto para renderizar la plantilla.

    Returns:
        HttpResponse: Una respuesta HTTP con el PDF o un mensaje de error.
    """
    # Carga la plantilla.
    template = get_template(template_src)
    # Renderiza la plantilla con el contexto, obteniendo el HTML.
    html = template.render(context_dict)
    # Crea un buffer en memoria para almacenar el PDF.
    result = io.BytesIO() 
    
    # Usa pisa para convertir el HTML (codificado en UTF-8) al PDF, guardándolo en 'result'.
    pdf = pisa.pisaDocument(io.BytesIO(html.encode("UTF-8")), result)
    
    # Si no hubo errores en la creación del PDF...
    if not pdf.err:
        # Devuelve una respuesta HTTP con el contenido del PDF y el tipo de contenido adecuado.
        return HttpResponse(result.getvalue(), content_type='application/pdf')
    # Si hubo un error, devuelve una respuesta HTTP con el mensaje de error.
    return HttpResponse(f"Error al generar PDF: {pdf.err}")


def download_pdf_report_view(request):
    """
    Vista que maneja la descarga del reporte de comparación en formato PDF.
    Recupera los datos de la sesión guardados previamente.
    """
    # Obtiene los datos del reporte guardados en la sesión.
    report_data = request.session.get('report_data')

    # Si no hay datos en la sesión, significa que el usuario no generó un reporte
    # o la sesión expiró, por lo que devuelve un error.
    if not report_data:
        return HttpResponse("No hay datos de informe para generar el PDF. Por favor, primero genera una comparación.", status=404)

    # Añade una bandera al contexto para que la plantilla PDF sepa que se está renderizando para PDF
    # (puede ser útil para estilos o contenido específico del PDF).
    report_data['is_pdf'] = True

    # Llama a la función 'render_to_pdf' para generar y devolver el PDF.
    return render_to_pdf('chapterThree/pdf_template_ch3.html', report_data)