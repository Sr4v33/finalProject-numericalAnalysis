# chapterThree/views.py

from django.shortcuts import render
from django.http import HttpRequest, HttpResponse, JsonResponse
from .forms import InterpolationForm, PointFormSet
from .methods import calculate_vandermonde, calculate_newton, calculate_lagrange, calculate_linear_spline, calculate_cubic_spline
import numpy as np
import json
import matplotlib.pyplot as plt
import io
import urllib



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
                    matrix_a, vector_b, coeffs, poly_str, plot_data = calculate_vandermonde(points_data)
                    context['results'] = {
                        'type': 'vandermonde', # Añadimos un tipo
                        'points': points_data,
                        'matrixA': matrix_a,
                        'B': vector_b,
                        'coeffs': [f"{c:.6f}" for c in coeffs],
                        'polynom': poly_str,
                        'plot_data': plot_data
                    }
                
                elif method == 'newton': # <-- AÑADIMOS NEWTON
                    headers, table, coeffs, poly_str, poly_str_newton_native, plot_data = calculate_newton(points_data)
                    context['results'] = {
                        'type': 'newton', # Añadimos un tipo
                        'points': points_data,
                        'headers': headers,
                        'table': table,
                        'coeffs': coeffs, # Ya es lista
                        'polynom': poly_str,
                        'polynom_newton_form': poly_str_newton_native,
                        'plot_data': plot_data
                    }
                
                elif method == 'lagrange': # <-- AÑADIMOS LAGRANGE
                    basis_table, native_poly, expanded_poly, plot_json = calculate_lagrange(points_data)
                    context['results'] = {
                        'type': 'lagrange',
                        'points': points_data,
                        'lagrange_basis_polynomials': basis_table,
                        'polynom_lagrange_form': native_poly,
                        'polynom': expanded_poly, # Para consistencia con el nombre de la forma expandida
                        'plot_data': plot_json
                    }

                elif method == 'spline_linear': # <-- NUEVO
                    coeffs_table, tracers, plot_json = calculate_linear_spline(points_data)
                    context['results'] = {
                        'type': 'spline_linear',
                        'points': points_data,
                        'coeffs_table': coeffs_table,
                        'tracers_list': tracers,
                        'plot_data': plot_json
                    }

                elif method == 'spline_cubic': # <-- NUEVO
                    coeffs_table, tracers, plot_json = calculate_cubic_spline(points_data)
                    context['results'] = {
                        'type': 'spline_cubic',
                        'points': points_data,
                        'coeffs_table': coeffs_table,
                        'tracers_list': tracers,
                        'plot_data': plot_json
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
    Vista para renderizar el gráfico con más puntos interpolados.
    """
    try:
        # Obtener datos originales
        original_x = json.loads(request.GET.get('original_x', '[]'))
        original_y = json.loads(request.GET.get('original_y', '[]'))
        function_string = request.GET.get('function_name', 'Interpolación')

        if not original_x or not original_y:
            raise ValueError("Faltan datos para la interpolación.")

        # Asegúrate de que tienen la misma longitud
        if len(original_x) != len(original_y):
            raise ValueError("Longitud de X y Y no coincide.")

        # Convertir a numpy arrays
        x = np.array(original_x)
        y = np.array(original_y)

        # Ajustar un polinomio (grado = len(x)-1 para interpolar)
        coeffs = np.polyfit(x, y, deg=len(x) - 1)
        poly = np.poly1d(coeffs)

        # Crear puntos para una línea suave
        min_x, max_x = min(x), max(x)
        interpolated_x = np.linspace(min_x - 50, max_x + 50, 1000).tolist()
        interpolated_y = poly(interpolated_x).tolist()

        plot_data = json.dumps({
            'original_x': original_x,
            'original_y': original_y,
            'interpolated_x': interpolated_x,
            'interpolated_y': interpolated_y,
        })

        context = {
            'function_string': function_string,
            'plot_data': plot_data,
            'error': None,
        }
        return render(request, 'chapterThree/graph_ch3.html', context)

    except (json.JSONDecodeError, ValueError) as e:
        context = {'error': f"Error al procesar los datos para el gráfico: {e}"}
        return render(request, 'chapterThree/graph_ch3.html', context)
    except Exception as e:
        context = {'error': f"Ocurrió un error inesperado: {e}"}
        return render(request, 'chapterThree/graph_ch3.html', context)


# Vista para comparar métodos (Placeholder)
def compare_methods_view(request: HttpRequest) -> HttpResponse:
    # Esta vista necesitaría una lógica similar a 'interpolation_view'
    # pero calculando para TODOS los métodos y presentándolos
    # en un template diferente o en la misma con una estructura adaptada.
    return HttpResponse("La comparación de métodos aún no está implementada.")