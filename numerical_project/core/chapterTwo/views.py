# Importaciones necesarias de Django.
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template.loader import get_template

# Importaciones de bibliotecas de terceros.
from xhtml2pdf import pisa  # Para la conversión de HTML a PDF.
import io                   # Para manejar flujos de bytes en memoria (PDF).
import numpy as np          # Para cálculos numéricos con matrices y vectores.
import json                 # Para serializar datos a JSON (aunque no se usa mucho aquí, es común).
import ast                  # Para evaluar de forma segura strings como literales de Python (listas, etc.).

# Importaciones locales (de esta aplicación Django).
from .forms import MatrixMethodForm, METHOD_CHOICES_CH2 # El formulario y las opciones de método.
from . import methods as ch2_methods                   # Los métodos de cálculo (Jacobi, GS, SOR).

# --- Funciones Auxiliares de Parseo y Validación ---

def parse_matrix_string(matrix_str, max_size=7):
    """
    Parsea (interpreta) una cadena de texto que representa una matriz
    y la convierte en un array de NumPy, validando su formato y tamaño.

    Args:
        matrix_str (str): La cadena de texto a parsear (ej: "[[1,2],[3,4]]").
        max_size (int): El tamaño máximo permitido para filas y columnas.

    Returns:
        tuple: (np.ndarray, None) si el parseo es exitoso,
               (None, str) si hay un error, donde str es el mensaje.
    """
    try:
        # Usa ast.literal_eval en lugar de eval() por seguridad.
        # Solo parsea literales de Python (listas, tuplas, números, etc.).
        parsed_list = ast.literal_eval(matrix_str)
        
        # Validación: ¿Es una lista de listas?
        if not isinstance(parsed_list, list) or not all(isinstance(row, list) for row in parsed_list):
            return None, "Formato de matriz inválido. Debe ser una lista de listas."
        
        rows = len(parsed_list)
        # Validación: ¿Está vacía?
        if rows == 0:
            return None, "La matriz A no puede estar vacía."
        # Validación: ¿Excede el tamaño máximo de filas?
        if rows > max_size:
            return None, f"La matriz A excede el tamaño máximo de {max_size}x{max_size}."
        
        cols = len(parsed_list[0])
        # Validación: ¿Excede el tamaño máximo de columnas?
        if cols > max_size:
            return None, f"La matriz A excede el tamaño máximo de {max_size}x{max_size}."

        # Validación: ¿Son todas las filas del mismo tamaño (es rectangular)?
        if not all(len(row) == cols for row in parsed_list):
            return None, "Todas las filas de la matriz A deben tener el mismo número de columnas."
        
        # Conversión a array de NumPy con tipo flotante.
        matrix = np.array(parsed_list, dtype=float)
        # Validación: ¿Es realmente 2D?
        if matrix.ndim != 2:
            return None, "Formato de matriz A inválido después de la conversión."
            
        # Si todo está bien, devuelve la matriz y None (sin error).
        return matrix, None
    # Captura errores de parseo (sintaxis) o de tipo.
    except (ValueError, SyntaxError, TypeError) as e:
        return None, f"Error al parsear matriz A: {e}. Asegúrate de usar un formato como [[1,2],[3,4]]."

def parse_vector_string(vector_str, expected_size=None, max_size=7):
    """
    Parsea (interpreta) una cadena de texto que representa un vector
    y la convierte en un array de NumPy, validando su formato y tamaño.

    Args:
        vector_str (str): La cadena de texto a parsear (ej: "[1,2,3]").
        expected_size (int, optional): El tamaño que debe tener el vector.
        max_size (int): El tamaño máximo permitido para el vector.

    Returns:
        tuple: (np.ndarray, None) si el parseo es exitoso,
               (None, str) si hay un error, donde str es el mensaje.
    """
    try:
        # Parsea de forma segura.
        parsed_list = ast.literal_eval(vector_str)
        # Validación: ¿Es una lista simple (no lista de listas)?
        if not isinstance(parsed_list, list) or any(isinstance(item, list) for item in parsed_list):
            return None, "Formato de vector inválido. Debe ser una lista simple. Ej: [1,2,3]"

        # Validación: ¿Excede el tamaño máximo?
        if len(parsed_list) > max_size:
            return None, f"El vector excede el tamaño máximo de {max_size}."

        # Conversión a array de NumPy.
        vector = np.array(parsed_list, dtype=float)
        # Validación: ¿Es realmente 1D?
        if vector.ndim != 1:
            return None, "Formato de vector inválido después de la conversión."
        
        # Validación: ¿Tiene el tamaño esperado (si se especificó)?
        if expected_size is not None and len(vector) != expected_size:
            return None, f"El vector debe tener un tamaño de {expected_size}."
            
        # Devuelve el vector y None (sin error).
        return vector, None
    # Captura errores de parseo o tipo.
    except (ValueError, SyntaxError, TypeError) as e:
        return None, f"Error al parsear vector: {e}. Asegúrate de usar un formato como [1,2,3]."

# --- Vistas de Django ---

def chapter_two_view(request):
    """
    Vista principal para el Capítulo 2. Muestra el formulario y procesa
    la ejecución de UN SOLO método iterativo seleccionado por el usuario.
    """
    # Inicializa el formulario. Si es POST, lo llena con datos; si es GET, lo deja vacío.
    form = MatrixMethodForm(request.POST or None)
    # Contexto base para la plantilla.
    context = {'form': form, 'page_title': "Capítulo 2: Sistemas de Ecuaciones Lineales"}

    # Si la petición es POST y el formulario es válido...
    if request.method == 'POST' and form.is_valid():
        # Obtiene los datos limpios del formulario.
        data = form.cleaned_data
        method_choice = data['method'] # El método elegido por el usuario.

        # Parsea la matriz A y maneja errores.
        A, error_A = parse_matrix_string(data['matrix_A_str'])
        if error_A:
            context['error_message'] = error_A
            return render(request, 'chapterTwo/chapterTwo.html', context)

        # Obtiene el tamaño 'n' y parsea los vectores b y x0, validando su tamaño.
        n = A.shape[0]
        b, error_b = parse_vector_string(data['vector_b_str'], expected_size=n)
        if error_b:
            context['error_message'] = error_b
            return render(request, 'chapterTwo/chapterTwo.html', context)

        x0, error_x0 = parse_vector_string(data['vector_x0_str'], expected_size=n)
        if error_x0:
            context['error_message'] = error_x0
            return render(request, 'chapterTwo/chapterTwo.html', context)

        # Obtiene los parámetros de iteración.
        tol = data['tolerance']
        niter = data['max_iterations']
        omega = data.get('omega_sor') # .get() es seguro si 'omega_sor' no está.

        # Ejecuta el método seleccionado.
        current_method_results = {}
        if method_choice == 'jacobi':
            current_method_results[method_choice] = ch2_methods.method_jacobi(A, b, x0.copy(), tol, niter)
        elif method_choice == 'gauss_seidel':
            current_method_results[method_choice] = ch2_methods.method_gauss_seidel(A, b, x0.copy(), tol, niter)
        elif method_choice == 'sor':
            current_method_results[method_choice] = ch2_methods.method_sor(A, b, x0.copy(), tol, niter, omega)
            
        # Si se ejecutó un método (y no es 'comparar'), prepara los resultados para la plantilla.
        if method_choice != 'compare_all_ch2' and method_choice in current_method_results:
            res_list, msg, spec_rad, conv_msg = current_method_results[method_choice]
            context.update({
                'results_table': res_list,
                'final_message': msg,
                'spectral_radius': spec_rad,
                'convergence_message': conv_msg,
                'method_name': dict(METHOD_CHOICES_CH2)[method_choice], # Nombre legible.
                'vector_size': n,
            })
        # Si se seleccionó 'comparar', esta vista no lo maneja (lo hace compare_methods_ch2_view).
        elif method_choice == 'compare_all_ch2':
            # Esto normalmente no debería ocurrir si los botones del formulario apuntan a la URL correcta.
            print("DEBUG: 'compare_all_ch2' was selected, should be handled by compare_methods_ch2_view.")

    # Si el formulario POST no es válido, añade los errores al contexto.
    elif request.method == 'POST' and not form.is_valid():
        context['error_message'] = "Errores en el formulario."
        context['form_errors'] = form.errors

    # Renderiza la plantilla principal del capítulo.
    return render(request, 'chapterTwo/chapterTwo.html', context)

def run_all_ch2_methods(A, b, x0, tol, niter, omega):
    """
    Función auxiliar para ejecutar todos los métodos (Jacobi, GS, SOR)
    y recopilar sus resultados.

    Returns:
        dict: Un diccionario donde las claves son los nombres de los métodos
              y los valores son tuplas con sus resultados.
    """
    all_method_outputs = {}
    
    # Ejecuta Jacobi y captura errores.
    try:
        res_list, msg, spec_rad, conv_msg = ch2_methods.method_jacobi(A, b, x0.copy(), tol, niter)
        # Guarda resultados y el número de iteraciones.
        all_method_outputs['Jacobi'] = (res_list, msg, spec_rad, conv_msg, len(res_list)-1 if res_list else niter)
    except Exception as e:
        all_method_outputs['Jacobi'] = ([], f"Error: {e}", np.inf, "No se pudo ejecutar.", niter)

    # Ejecuta Gauss-Seidel y captura errores.
    try:
        res_list, msg, spec_rad, conv_msg = ch2_methods.method_gauss_seidel(A, b, x0.copy(), tol, niter)
        all_method_outputs['Gauss-Seidel'] = (res_list, msg, spec_rad, conv_msg, len(res_list)-1 if res_list else niter)
    except Exception as e:
        all_method_outputs['Gauss-Seidel'] = ([], f"Error: {e}", np.inf, "No se pudo ejecutar.", niter)
        
    # Ejecuta SOR solo si se proporcionó omega.
    if omega is not None:
        try:
            res_sor, msg_sor, spec_rad_sor, conv_msg_sor = ch2_methods.method_sor(A, b, x0.copy(), tol, niter, omega)
            all_method_outputs['SOR'] = (res_sor, msg_sor, spec_rad_sor, conv_msg_sor, len(res_sor)-1 if res_sor else niter)
        except Exception as e:
            all_method_outputs['SOR'] = ([], f"Error en SOR: {e} (ω={omega})", np.inf, "No se pudo ejecutar.", niter)
    else:
        all_method_outputs['SOR'] = ([], "Omega (ω) no proporcionado para SOR.", np.inf, "Requiere ω.", niter)
        
    return all_method_outputs

def find_best_ch2_method(all_method_outputs, tol):
    """
    Analiza los resultados de todos los métodos y determina cuál es el "mejor"
    según criterios de convergencia y número de iteraciones.

    Args:
        all_method_outputs (dict): El diccionario devuelto por run_all_ch2_methods.
        tol (float): La tolerancia usada, para verificar convergencia.

    Returns:
        tuple: (nombre_mejor_metodo, min_iteraciones) o (None, None).
    """
    best_method_name = None
    min_iterations = float('inf')
    min_error_at_min_iterations = float('inf')
    best_method_priority = 3 # Prioridad: 1 (converge ρ<1 y tol), 2 (solo tol), 3 (ninguno).

    for name, (res_list, msg, spec_rad, conv_msg, iterations) in all_method_outputs.items():
        current_priority = 3 # Asume la peor prioridad.
        
        # ¿Converge teóricamente (ρ<1)?
        converged_by_rho = False
        if not np.isinf(spec_rad) and not np.isnan(spec_rad):
            converged_by_rho = spec_rad < 1

        # ¿Converge según la tolerancia?
        converged_by_tol = False
        final_error = float('inf')
        if res_list and 'Error' in res_list[-1]:
            error_val = res_list[-1]['Error']
            if isinstance(error_val, (int, float)) and not np.isnan(error_val): # Asegura que sea un número válido.
                final_error = error_val
                if final_error <= tol:
                    converged_by_tol = True
        
        # Asigna prioridad actual basada en convergencia.
        if converged_by_rho and converged_by_tol:
            current_priority = 1
        elif converged_by_tol:
            current_priority = 2
        else: # Si no converge, no puede ser el mejor.
            continue

        # Compara con el mejor encontrado hasta ahora.
        if current_priority < best_method_priority: # Encontró uno de mayor prioridad.
            best_method_priority = current_priority
            min_iterations = iterations
            min_error_at_min_iterations = final_error
            best_method_name = name
        elif current_priority == best_method_priority: # Misma prioridad, decide por iteraciones.
            if iterations < min_iterations:
                min_iterations = iterations
                min_error_at_min_iterations = final_error
                best_method_name = name
            elif iterations == min_iterations: # Mismas iteraciones, decide por error.
                if final_error < min_error_at_min_iterations:
                    min_error_at_min_iterations = final_error
                    best_method_name = name
    
    return best_method_name, min_iterations if best_method_name else None

def compare_methods_ch2_view(request):
    """
    Vista para mostrar el reporte de comparación de todos los métodos.
    Se accede vía POST desde el formulario principal cuando se elige 'Comparar Todos'.
    """
    form = MatrixMethodForm(request.POST or None) # Acepta POST, pero si se accede por GET, muestra vacío.
    context = {'form': form, 'page_title': "Capítulo 2: Comparación de Métodos"}

    # Solo procesa si es POST y el formulario es válido.
    if request.method == 'POST' and form.is_valid():
        data = form.cleaned_data
        
        # Parsea A, b y x0, manejando errores.
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

        # Ejecuta todos los métodos y encuentra el mejor.
        all_method_outputs = run_all_ch2_methods(A, b, x0, tol, niter, omega)
        best_method, min_iter = find_best_ch2_method(all_method_outputs, tol)

        # Prepara el contexto para la plantilla de comparación.
        context['all_method_outputs'] = all_method_outputs
        context['best_method'] = best_method
        context['min_iter'] = min_iter
        context['input_data'] = { # Guarda los datos de entrada para mostrarlos.
            'A_str': data['matrix_A_str'], 'b_str': data['vector_b_str'],
            'x0_str': data['vector_x0_str'], 'tol': tol, 'niter': niter, 'omega': omega,
        }
        context['vector_size'] = n

        # Guarda los resultados en la sesión para poder generar el PDF después.
        # Se convierte ρ(T) a 'inf' si es np.inf para que sea serializable.
        request.session['report_data_ch2'] = {
            'all_method_outputs': {k: (v[0], v[1], float(v[2]) if not np.isinf(v[2]) else 'inf', v[3], v[4]) for k, v in all_method_outputs.items()},
            'best_method': best_method, 'min_iter': min_iter,
            'input_data': context['input_data'], 'vector_size': n,
        }
    # Si hay errores en el POST.
    elif request.method == 'POST' and not form.is_valid():
        context['form_errors'] = form.errors

    # Renderiza la plantilla del reporte de comparación.
    return render(request, 'chapterTwo/compare_report_ch2.html', context)

# --- Funciones para Generar PDF ---

def render_to_pdf_ch2(template_src, context_dict={}):
    """
    Función auxiliar genérica para renderizar una plantilla HTML a PDF
    usando xhtml2pdf.
    """
    template = get_template(template_src)
    html = template.render(context_dict)
    result = io.BytesIO() # Buffer en memoria.

    # Llama a pisa para crear el PDF.
    pdf = pisa.pisaDocument(io.BytesIO(html.encode("UTF-8")), result)
    # Si no hay error, devuelve el PDF como HttpResponse.
    if not pdf.err:
        return HttpResponse(result.getvalue(), content_type='application/pdf')
    # Si hay error, devuelve un mensaje de error.
    return HttpResponse(f"Error al generar PDF: {pdf.err}")

def download_pdf_report_ch2_view(request):
    """
    Vista para manejar la descarga del reporte de comparación en PDF.
    Obtiene los datos guardados en la sesión.
    """
    # Recupera los datos de la sesión.
    report_data = request.session.get('report_data_ch2')
    # Si no hay datos, muestra un error.
    if not report_data:
        return HttpResponse("No hay datos para el PDF. Por favor, genera un reporte primero.", status=404)
    # Añade una bandera para indicar que se está renderizando como PDF.
    report_data['is_pdf'] = True
    # Llama a la función de renderizado a PDF.
    return render_to_pdf_ch2('chapterTwo/pdf_template_ch2.html', report_data)