from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template.loader import get_template
from .forms import MatrixMethodForm, METHOD_CHOICES_CH2
from . import methods as ch2_methods
from xhtml2pdf import pisa 
import io
import numpy as np
import json # Para convertir Python lists/dicts a JSON
import ast 

from .forms import MatrixMethodForm
from . import methods as ch2_methods

def parse_matrix_string(matrix_str, max_size=7):
    """Parsea un string a una matriz numpy, validando tamaño."""
    try:
        # Usar ast.literal_eval para parsear de forma segura
        parsed_list = ast.literal_eval(matrix_str)
        # Validaciones básicas
        if not isinstance(parsed_list, list) or not all(isinstance(row, list) for row in parsed_list):
            return None, "Formato de matriz inválido. Debe ser una lista de listas."
        
        rows = len(parsed_list)
        if rows == 0:
            return None, "La matriz A no puede estar vacía."
        if rows > max_size:
            return None, f"La matriz A excede el tamaño máximo de {max_size}x{max_size}."
        
        cols = len(parsed_list[0])
        if cols > max_size:
             return None, f"La matriz A excede el tamaño máximo de {max_size}x{max_size}."

        if not all(len(row) == cols for row in parsed_list):
            return None, "Todas las filas de la matriz A deben tener el mismo número de columnas."
        
        # Convertir a numpy array de flotantes
        matrix = np.array(parsed_list, dtype=float)
        if matrix.ndim != 2: # Asegurar que es 2D
            return None, "Formato de matriz A inválido después de la conversión."
        return matrix, None
    except (ValueError, SyntaxError, TypeError) as e:
        return None, f"Error al parsear matriz A: {e}. Asegúrate de usar un formato como [[1,2],[3,4]]."

def parse_vector_string(vector_str, expected_size=None, max_size=7):
    """Parsea un string a un vector numpy, validando tamaño."""
    try:
        parsed_list = ast.literal_eval(vector_str)
        if not isinstance(parsed_list, list) or any(isinstance(item, list) for item in parsed_list):
             return None, "Formato de vector inválido. Debe ser una lista simple. Ej: [1,2,3]"

        if len(parsed_list) > max_size :
             return None, f"El vector excede el tamaño máximo de {max_size}."

        vector = np.array(parsed_list, dtype=float)
        if vector.ndim != 1: # Asegurar que es 1D
            return None, "Formato de vector inválido después de la conversión."
        
        if expected_size is not None and len(vector) != expected_size:
            return None, f"El vector debe tener un tamaño de {expected_size}."
        return vector, None
    except (ValueError, SyntaxError, TypeError) as e:
        return None, f"Error al parsear vector: {e}. Asegúrate de usar un formato como [1,2,3]."

# core/chapterTwo/views.py
def chapter_two_view(request):
    form = MatrixMethodForm(request.POST or None) # Se inicializa con POST si existe, o vacío si es GET
    context = {'form': form, 'page_title': "Capítulo 2: Sistemas de Ecuaciones Lineales"}

    # Imprime si el form es el que se envía o uno nuevo para GET
    print(f"DEBUG: chapter_two_view - Request method: {request.method}")
    if request.method == 'POST':
        print(f"DEBUG: POST data received: {request.POST}")

    if request.method == 'POST' and form.is_valid():
        print("DEBUG: Form is valid.") # CONFIRMA QUE EL FORM ES VÁLIDO
        data = form.cleaned_data
        method_choice = data['method']
        print(f"DEBUG: Method choice: {method_choice}") # VERIFICA EL MÉTODO

        # ... (parseo de A, b, x0, que ya sabemos que el JS los llena) ...
        # Asegúrate de que no haya errores aquí que impidan continuar.
        # Si hay error de parseo, se debería mostrar context['error_message'].

        A, error_A = parse_matrix_string(data['matrix_A_str'])
        if error_A:
            context['error_message'] = error_A
            print(f"DEBUG: Error parsing A: {error_A}")
            return render(request, 'chapterTwo/chapterTwo.html', context)
        print(f"DEBUG: Parsed A: {A}")

        n = A.shape[0]
        b, error_b = parse_vector_string(data['vector_b_str'], expected_size=n)
        if error_b:
            context['error_message'] = error_b
            print(f"DEBUG: Error parsing b: {error_b}")
            return render(request, 'chapterTwo/chapterTwo.html', context)
        print(f"DEBUG: Parsed b: {b}")

        x0, error_x0 = parse_vector_string(data['vector_x0_str'], expected_size=n)
        if error_x0:
            context['error_message'] = error_x0
            print(f"DEBUG: Error parsing x0: {error_x0}")
            return render(request, 'chapterTwo/chapterTwo.html', context)
        print(f"DEBUG: Parsed x0: {x0}")

        tol = data['tolerance']
        niter = data['max_iterations']
        omega = data.get('omega_sor') 
        print(f"DEBUG: tol={tol}, niter={niter}, omega={omega}")

        current_method_results = {}
        if method_choice == 'jacobi': # 'jacobi' es minúscula aquí
            print("DEBUG: Running Jacobi...")
                # Usa method_choice (que es 'jacobi') como clave
            current_method_results[method_choice] = ch2_methods.method_jacobi(A, b, x0.copy(), tol, niter)
        elif method_choice == 'gauss_seidel': # 'gauss_seidel' es minúscula
            print("DEBUG: Running Gauss-Seidel...")
                # Usa method_choice como clave
            current_method_results[method_choice] = ch2_methods.method_gauss_seidel(A, b, x0.copy(), tol, niter)
        elif method_choice == 'sor': # 'sor' es minúscula
            print("DEBUG: Running SOR...")
                # Usa method_choice como clave
            current_method_results[method_choice] = ch2_methods.method_sor(A, b, x0.copy(), tol, niter, omega)
            
        print(f"DEBUG: current_method_results: {current_method_results}")

        if method_choice != 'compare_all_ch2' and method_choice in current_method_results:
            print(f"DEBUG: Preparing context for single method: {method_choice}")
            res_list, msg, spec_rad, conv_msg = current_method_results[method_choice]
            print(f"DEBUG: res_list: {res_list}") # MIRA SI res_list TIENE DATOS
            context.update({
                'results_table': res_list,
                'final_message': msg,
                'spectral_radius': spec_rad,
                'convergence_message': conv_msg,
                'method_name': dict(METHOD_CHOICES_CH2)[method_choice],
                'vector_size': n,
            })
        elif method_choice == 'compare_all_ch2':
             print("DEBUG: 'compare_all_ch2' was selected, this should be handled by compare_methods_ch2_view.")
             # No debería llegar aquí si el formaction es correcto para 'compare_all_ch2'
        else:
            print(f"DEBUG: No specific method results to add to context for method_choice: {method_choice}")

    elif request.method == 'POST' and not form.is_valid():
        print(f"DEBUG: Form is INVALID. Errors: {form.errors}")
        context['error_message'] = "Errores en el formulario."
        context['form_errors'] = form.errors

    return render(request, 'chapterTwo/chapterTwo.html', context)





def run_all_ch2_methods(A, b, x0, tol, niter, omega):
    """Función auxiliar para ejecutar todos los métodos del Cap 2."""
    all_method_outputs = {}
    
    # Jacobi
    try:
        res_list, msg, spec_rad, conv_msg = ch2_methods.method_jacobi(A, b, x0.copy(), tol, niter)
        all_method_outputs['Jacobi'] = (res_list, msg, spec_rad, conv_msg, len(res_list)-1 if res_list else niter)
    except Exception as e:
        all_method_outputs['Jacobi'] = ([], f"Error: {e}", np.inf, "No se pudo ejecutar.", niter)

    # Gauss-Seidel
    try:
        res_list, msg, spec_rad, conv_msg = ch2_methods.method_gauss_seidel(A, b, x0.copy(), tol, niter)
        all_method_outputs['Gauss-Seidel'] = (res_list, msg, spec_rad, conv_msg, len(res_list)-1 if res_list else niter)
    except Exception as e:
        all_method_outputs['Gauss-Seidel'] = ([], f"Error: {e}", np.inf, "No se pudo ejecutar.", niter)
        
    # SOR
    if omega is not None: # Solo ejecutar SOR si omega tiene un valor
        try:
            res_sor, msg_sor, spec_rad_sor, conv_msg_sor = ch2_methods.method_sor(A, b, x0.copy(), tol, niter, omega)
            all_method_outputs['SOR'] = (res_sor, msg_sor, spec_rad_sor, conv_msg_sor, len(res_sor)-1 if res_sor else niter)
        except Exception as e:
            all_method_outputs['SOR'] = ([], f"Error en SOR: {e} (ω={omega})", np.inf, "No se pudo ejecutar.", niter)
    else:
        all_method_outputs['SOR'] = ([], "Omega (ω) no proporcionado para SOR.", np.inf, "Requiere ω.", niter)
        
    return all_method_outputs

def find_best_ch2_method(all_method_outputs, tol):
    best_method_name = None
    min_iterations = float('inf')
    
    for name, (res_list, msg, spec_rad, conv_msg, iterations) in all_method_outputs.items():
        converged_by_rho = spec_rad < 1
        converged_by_tol = False
        if res_list:
            final_error = res_list[-1]['Error']
            if not np.isnan(final_error) and final_error <= tol:
                converged_by_tol = True
        
        # Prioridad: ρ(T) < 1 y convergió por tolerancia
        if converged_by_rho and converged_by_tol:
            if iterations < min_iterations:
                min_iterations = iterations
                best_method_name = name
        # Segunda prioridad: Solo convergió por tolerancia (pero ρ(T) >= 1)
        elif converged_by_tol and best_method_name is None: # Solo si no hay uno "mejor" aún
             if iterations < min_iterations:
                min_iterations = iterations
                best_method_name = name # Podría ser sobreescrito por uno con rho<1
    
    return best_method_name, min_iterations if best_method_name else None


def compare_methods_ch2_view(request):
    form = MatrixMethodForm(request.POST or None)
    context = {'form': form, 'page_title': "Capítulo 2: Comparación de Métodos"}

    if request.method == 'POST' and form.is_valid():
        data = form.cleaned_data
        A, error_A = parse_matrix_string(data['matrix_A_str'])
        if error_A:
            context['global_error'] = error_A
            return render(request, 'chapterTwo/compare_report_ch2.html', context)

        n = A.shape[0]
        b, error_b = parse_vector_string(data['vector_b_str'], expected_size=n)
        if error_b:
            context['global_error'] = error_b
            return render(request, 'chapterTwo/compare_report_ch2.html', context)

        x0, error_x0 = parse_vector_string(data['vector_x0_str'], expected_size=n)
        if error_x0:
            context['global_error'] = error_x0
            return render(request, 'chapterTwo/compare_report_ch2.html', context)
            
        tol = data['tolerance']
        niter = data['max_iterations']
        omega = data.get('omega_sor')

        all_method_outputs = run_all_ch2_methods(A, b, x0, tol, niter, omega)
        best_method, min_iter = find_best_ch2_method(all_method_outputs, tol)

        context['all_method_outputs'] = all_method_outputs
        context['best_method'] = best_method
        context['min_iter'] = min_iter
        context['input_data'] = {
            'A_str': data['matrix_A_str'],
            'b_str': data['vector_b_str'],
            'x0_str': data['vector_x0_str'],
            'tol': tol,
            'niter': niter,
            'omega': omega,
        }
        context['vector_size'] = n

        request.session['report_data_ch2'] = {
            'all_method_outputs': {k: (v[0], v[1], float(v[2]) if not np.isinf(v[2]) else 'inf', v[3], v[4]) for k, v in all_method_outputs.items()}, # Serializar ρ
            'best_method': best_method,
            'min_iter': min_iter,
            'input_data': context['input_data'],
            'vector_size': n,
        }
    else: # Si el form no es válido o es GET
        if request.method == 'POST' and not form.is_valid():
            context['form_errors'] = form.errors


    return render(request, 'chapterTwo/compare_report_ch2.html', context)


def render_to_pdf_ch2(template_src, context_dict={}):
    template = get_template(template_src)
    html = template.render(context_dict)
    result = io.BytesIO()

    pdf = pisa.pisaDocument(io.BytesIO(html.encode("UTF-8")), result)
    if not pdf.err:
        return HttpResponse(result.getvalue(), content_type='application/pdf')
    return HttpResponse(f"Error al generar PDF: {pdf.err}")
    return HttpResponse(f"Generación de PDF deshabilitada. Contenido HTML:<br>{html}") # Placeholder

def download_pdf_report_ch2_view(request):
    report_data = request.session.get('report_data_ch2')
    if not report_data:
        return HttpResponse("No hay datos para el PDF.", status=404)
    report_data['is_pdf'] = True
    return render_to_pdf_ch2('chapterTwo/pdf_template_ch2.html', report_data)