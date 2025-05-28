# chapterThree/views.py

from django.shortcuts import render
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.template.loader import get_template
from .forms import InterpolationForm, PointFormSet
from .methods import (
    calculate_vandermonde, calculate_newton, calculate_lagrange,
    calculate_linear_spline, calculate_cubic_spline, evaluate_interpolation
)
from xhtml2pdf import pisa 
import numpy as np
import json
import matplotlib.pyplot as plt
import io







def interpolation_view(request: HttpRequest) -> HttpResponse:
    """Vista principal para la interpolación."""
    context = {
        'page_title': "Capítulo 3: Interpolación",
        'form': InterpolationForm(),
        'formset': PointFormSet(initial=[{'x': '', 'y': ''}, {'x': '', 'y': ''}]), # 2 filas iniciales
        'results': None,
        'error_message': None,
    }

    if request.method == "POST":
        form = InterpolationForm(request.POST)
        formset = PointFormSet(request.POST)
        context['form'] = form
        context['formset'] = formset

        if form.is_valid() and formset.is_valid():
            method = form.cleaned_data['method']
            points_data = [
                data for data in formset.cleaned_data if data and not data.get('DELETE')
            ]
            
            context['method_name'] = dict(form.fields['method'].choices).get(method, method)

            try:
                if method == 'vandermonde':
                    matrix_a, vector_b, coeffs, poly_str_normal, plot_data, poly_sympy, poly_str_latex = calculate_vandermonde(points_data)
                    context['results'] = {
                        'type': 'vandermonde', # Añadimos un tipo
                        'points': points_data,
                        'matrixA': matrix_a,
                        'B': vector_b,
                        'coeffs': [f"{c:.6f}" for c in coeffs],
                        'polynom': poly_str_normal,
                        'plot_data': plot_data
                    }
                
                elif method == 'newton': # <-- AÑADIMOS NEWTON
                    headers, display_table, newton_coeffs_divided_differences, poly_str_expanded, poly_str_newton_native, plot_data, poly_sympy, poly_str_latex = calculate_newton(points_data)
                    context['results'] = {
                        'type': 'newton', # Añadimos un tipo
                        'points': points_data,
                        'headers': headers,
                        'table': display_table,
                        'coeffs': newton_coeffs_divided_differences,
                        'polynom': poly_str_expanded,
                        'polynom_newton_form': poly_str_newton_native,
                        'plot_data': plot_data
                    }
                
                elif method == 'lagrange': # <-- AÑADIMOS LAGRANGE
                    lagrange_basis_for_table, poly_str_lagrange_native, poly_str_expanded, plot_data, poly_sympy, poly_str_latex = calculate_lagrange(points_data)
                    context['results'] = {
                        'type': 'lagrange',
                        'points': points_data,
                        'lagrange_basis_polynomials': lagrange_basis_for_table,
                        'polynom_lagrange_form': poly_str_lagrange_native,
                        'polynom': poly_str_expanded, # Para consistencia con el nombre de la forma expandida
                        'plot_data': plot_data
                    }

                elif method == 'spline_linear':
                    coeffs_table_list, tracers_list, plot_data, spline_sympy_list, poly_str_latex = calculate_linear_spline(points_data)

                    polynom_parts = [f"S_{{{item['i']}}}(x) = {item['tracer_str']}" for item in tracers_list]

                    latex_lines = []
                    for item in tracers_list:

                        latex_lines.append(f"S_{{{item['i']}}}(x) = {item['tracer_str']}")

                    polynom_string = "; \\; ".join(latex_lines)
                    
                    context['results'] = {
                        'type': 'spline_linear',
                        'points': points_data,
                        'coeffs_table': coeffs_table_list,
                        'tracers_list': tracers_list,
                        'plot_data': plot_data,
                        'polynom': polynom_string
                    }
                
                elif method == 'spline_cubic': # <-- AÑADIR BLOQUE
                    coeffs_table_list, tracers_list, plot_data, spline_sympy_list, poly_str_latex = calculate_cubic_spline(points_data)

                    polynom_parts = [f"S_{{{item['i']}}}(x) = {item['tracer_str']}" for item in tracers_list]

                    latex_lines = []
                    for item in tracers_list:

                        latex_lines.append(f"S_{{{item['i']}}}(x) = {item['tracer_str']}")
                        latex_lines.append("\\\\")

                    polynom_string = "; \\; ".join(latex_lines)
                    
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

            except (np.linalg.LinAlgError, ValueError) as e:
                context['error_message'] = f"Error al calcular: {e}"
            except Exception as e:
                context['error_message'] = f"Ocurrió un error inesperado: {e}"

    return render(request, 'chapterThree/chapterThree.html', context)



def graph_view(request: HttpRequest) -> HttpResponse:
    """
    Vista para renderizar el gráfico, priorizando plot_data si se pasa.
    """
    try:
        # INTENTA OBTENER EL PLOT_DATA COMPLETO PRIMERO
        plot_data_str = request.GET.get('plot_data', None)
        function_string = request.GET.get('function_name', 'Interpolación')

        if plot_data_str:
            # Si se pasó plot_data, ¡úsalo directamente!
            # Valida que sea un JSON válido por seguridad (opcional pero recomendado)
            try:
                json.loads(plot_data_str)
                plot_data_to_render = plot_data_str
            except json.JSONDecodeError:
                raise ValueError("El 'plot_data' recibido no es un JSON válido.")

        else:
            # FALLBACK: Si no se pasó plot_data, intenta calcular (como antes)
            # PERO: Este fallback solo funciona bien para POLINOMIOS.
            # Para que funcione con splines, necesitarías pasar 'method_type'
            # y recalcular aquí, lo cual es menos ideal.
            # Nos enfocaremos en que el frontend *siempre* pase plot_data.
            # Si quieres un fallback robusto, tendrías que añadir la lógica
            # con 'method_type' como discutimos.

            original_x = json.loads(request.GET.get('original_x', '[]'))
            original_y = json.loads(request.GET.get('original_y', '[]'))

            if not original_x or not original_y:
                raise ValueError("Faltan datos para la interpolación (fallback).")
            if len(original_x) != len(original_y):
                raise ValueError("Longitud de X y Y no coincide (fallback).")

            x = np.array(original_x)
            y = np.array(original_y)

            # Ajustar un polinomio (grado = len(x)-1 para interpolar)
            coeffs = np.polyfit(x, y, deg=len(x) - 1)
            poly = np.poly1d(coeffs)

            # Crear puntos para una línea suave
            min_x, max_x = min(x), max(x)
            interpolated_x = np.linspace(min_x - 50, max_x + 50, 1000).tolist()
            interpolated_y = poly(interpolated_x).tolist()

            plot_data_to_render = json.dumps({
                'original_x': original_x,
                'original_y': original_y,
                'interpolated_x': interpolated_x,
                'interpolated_y': interpolated_y,
            })

        # Prepara el contexto y renderiza
        context = {
            'function_string': function_string,
            'plot_data': plot_data_to_render, # Usamos la variable unificada
            'error': None,
        }
        return render(request, 'chapterThree/graph_ch3.html', context)

    except (json.JSONDecodeError, ValueError) as e:
        context = {'error': f"Error al procesar los datos para el gráfico: {e}"}
        return render(request, 'chapterThree/graph_ch3.html', context)
    except Exception as e:
        context = {'error': f"Ocurrió un error inesperado: {e}"}
        return render(request, 'chapterThree/graph_ch3.html', context)

def compare_methods_view(request: HttpRequest) -> HttpResponse:
    context = {
        'page_title': "Reporte de Comparación de Métodos de Interpolación",
        'form': InterpolationForm(),
        'formset': PointFormSet(),
    }

    if request.method != "POST":
        context['error_message'] = "Acceso inválido a la comparación. Por favor, usa el formulario."
        return render(request, 'chapterThree/chapterThree.html', context)

    form = InterpolationForm(request.POST)
    formset = PointFormSet(request.POST)

    if form.is_valid() and formset.is_valid():
        points_data = [
            data for data in formset.cleaned_data if data and not data.get('DELETE')
        ]
        x_eval = form.cleaned_data.get('x_eval')
        y_eval = form.cleaned_data.get('y_eval')

        if x_eval is None or y_eval is None:
            error_context = {
                'page_title': "Capítulo 3: Interpolación",
                'form': form,
                'formset': formset,
                'error_message': "Para comparar métodos, debes ingresar un valor para 'X para Evaluar Error' y 'Y Real en X_eval'.",
            }
            return render(request, 'chapterThree/chapterThree.html', error_context)

        # Datos que irán tanto al contexto HTML como a la sesión para el PDF
        report_content_data = {
            'x_eval_report': x_eval,
            'y_eval_report': y_eval,
            'points_data_report': points_data,
            'comparison_results': [], # Se llenará a continuación
            'best_method': None,    # Se llenará a continuación
            'page_title': "Reporte de Comparación de Métodos de Interpolación" # Título para el PDF
        }

        methods_to_run = {
            "Vandermonde": calculate_vandermonde,
            "Newton": calculate_newton,
            "Lagrange": calculate_lagrange,
            "Spline Lineal": calculate_linear_spline,
            "Spline Cúbico": calculate_cubic_spline,
        }

        comparison_results_list = []

        for method_name, calculate_function in methods_to_run.items():
            poly_data_for_eval = None
            representative_poly_str = "N/A"
            
            try:
                if method_name == "Vandermonde":
                    _, _, _, _, _, poly_data_for_eval, representative_poly_str = calculate_function(points_data)
                elif method_name == "Newton":
                    _, _, _, _, _, _, poly_data_for_eval, representative_poly_str = calculate_function(points_data)
                elif method_name == "Lagrange":
                    _, _, _, _, poly_data_for_eval, representative_poly_str = calculate_function(points_data)
                elif method_name in ["Spline Lineal", "Spline Cúbico"]: # Asumiendo que devuelven lo mismo
                    _, _, _, poly_data_for_eval, representative_poly_str = calculate_function(points_data)
                
                y_predicted = evaluate_interpolation(poly_data_for_eval, x_eval)
                
                if np.isnan(y_predicted):
                    error = float('inf')
                    y_predicted_display = "Error (NaN)"
                else:
                    error = abs(y_eval - y_predicted)
                    y_predicted_display = f"{y_predicted:.6f}"

                comparison_results_list.append({
                    'name': method_name,
                    'poly_str': representative_poly_str,
                    'y_predicted': y_predicted_display,
                    'error': error
                })
            except ValueError as ve:
                print(f"Error de validación calculando {method_name}: {ve}")
                comparison_results_list.append({
                    'name': method_name, 'poly_str': f"Error: {ve}",
                    'y_predicted': "Error", 'error': float('inf')
                })
            except Exception as e:
                print(f"Error general calculando {method_name}: {e}")
                comparison_results_list.append({
                    'name': method_name, 'poly_str': "Error en cálculo",
                    'y_predicted': "Error", 'error': float('inf')
                })
        
        report_content_data['comparison_results'] = comparison_results_list
        
        valid_results = [res for res in comparison_results_list if isinstance(res['error'], (int, float)) and res['error'] != float('inf')]
        if valid_results:
            report_content_data['best_method'] = min(valid_results, key=lambda x: x['error'])

        # === GUARDAR DATOS PARA EL PDF EN LA SESIÓN ===
        request.session['report_data'] = report_content_data
        # ==============================================
        
        # Prepara el contexto para la plantilla HTML (que es el mismo que se guardó para el PDF)
        context.update(report_content_data)
        
        return render(request, 'chapterThree/compare_report_ch3.html', context)

    else:
        context['form'] = form
        context['formset'] = formset
        context['error_message'] = "Hubo errores en el formulario. Por favor, revisa los campos."
        return render(request, 'chapterThree/chapterThree.html', context)

    


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

    return render_to_pdf('chapterThree/pdf_template_ch3.html', report_data)