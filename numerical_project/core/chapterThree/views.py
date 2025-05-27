# chapterThree/views.py

from django.shortcuts import render
from django.http import HttpRequest, HttpResponse, JsonResponse
from .forms import InterpolationForm, PointFormSet
from .methods import calculate_vandermonde
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
            # Obtener solo los puntos que no están marcados para eliminar
            points_data = [
                data for data in formset.cleaned_data if data and not data.get('DELETE')
            ]
            
            context['method_name'] = dict(form.fields['method'].choices).get(method, method)

            if method == 'vandermonde':
                try:
                    matrix_a, vector_b, coeffs, poly_str, plot_data = calculate_vandermonde(points_data)
                    context['results'] = {
                        'points': points_data,
                        'matrixA': matrix_a,
                        'B': vector_b,
                        'coeffs': [f"{c:.6f}" for c in coeffs], # Formatear para mostrar
                        'polynom': poly_str,
                        'plot_data': plot_data # Ya está en formato JSON string
                    }
                except (np.linalg.LinAlgError, ValueError) as e:
                    context['error_message'] = f"Error al calcular Vandermonde: {e}"
                except Exception as e:
                    context['error_message'] = f"Ocurrió un error inesperado: {e}"

            # Aquí podrías añadir bloques elif para otros métodos:
            # elif method == 'newton':
            #     # ... llamar a calculate_newton ...
            #     pass

            else:
                context['error_message'] = f"El método '{method}' aún no está implementado."

        # Si los formularios no son válidos, los errores se mostrarán
        # automáticamente a través del template, ya que pasamos
        # `form` y `formset` con los datos POST y sus errores.

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