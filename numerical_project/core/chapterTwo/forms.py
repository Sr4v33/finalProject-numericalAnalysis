# Importa el módulo de formularios de Django.
from django import forms

# Define las opciones disponibles para los métodos de resolución de sistemas de ecuaciones.
# Cada tupla contiene el valor interno (usado en el código) y la etiqueta visible para el usuario.
METHOD_CHOICES_CH2 = [
    ('jacobi', 'Jacobi'),                         # Opción para el método de Jacobi.
    ('gauss_seidel', 'Gauss-Seidel'),             # Opción para el método de Gauss-Seidel.
    ('sor', 'SOR (Successive Over-Relaxation)'),  # Opción para el método SOR.
    ('compare_all_ch2', 'Comparar Todos los Métodos'), # Opción para ejecutar y comparar todos.
]

class MatrixMethodForm(forms.Form):
    """
    Formulario para ingresar los datos necesarios para resolver un sistema
    de ecuaciones lineales (Ax = b) usando métodos iterativos.
    """
    
    # Campo para ingresar la Matriz A como una cadena de texto.
    matrix_A_str = forms.CharField(
        label='Matriz A (Max 7x7)', # Etiqueta visible en el formulario.
        # Usa un widget Textarea (caja de texto multilínea) para facilitar la entrada.
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': 'Ej: [[4,1,0],[1,5,1],[0,1,6]]'}),
        # Texto de ayuda que guía al usuario sobre el formato esperado.
        help_text="Ingresa como lista de listas Python. Ej: [[1,2],[3,4]]",
        # No es requerido a nivel de formulario básico (la validación real puede ocurrir en la vista o clean).
        required=False 
    )
    
    # Campo para ingresar el Vector b como una cadena de texto.
    vector_b_str = forms.CharField(
        label='Vector b', # Etiqueta visible.
        # Usa un widget TextInput (caja de texto de una línea).
        widget=forms.TextInput(attrs={'placeholder': 'Ej: [7,8,9]'}),
        help_text="Ingresa como lista Python. Ej: [1,2]", # Texto de ayuda.
        required=False # No es requerido a nivel de formulario básico.
    )
    
    # Campo para ingresar el Vector inicial x0 como una cadena de texto.
    vector_x0_str = forms.CharField(
        label='Vector inicial x0', # Etiqueta visible.
        widget=forms.TextInput(attrs={'placeholder': 'Ej: [0,0,0]'}), # Widget y placeholder.
        help_text="Ingresa como lista Python. Ej: [0,0]", # Texto de ayuda.
        required=False # No es requerido a nivel de formulario básico.
    )
    
    # Campo numérico (flotante) para la tolerancia del error.
    tolerance = forms.FloatField(label='Tolerancia', initial=1e-7)
    
    # Campo numérico (entero) para el número máximo de iteraciones.
    max_iterations = forms.IntegerField(label='Máx. Iteraciones', initial=100)
    
    # Campo numérico (flotante) para el factor de relajación ω (omega), usado en SOR.
    omega_sor = forms.FloatField(
        label='Factor de Relajación ω (para SOR y Comparar Todos)', # Etiqueta visible.
        required=False, # No siempre es requerido, depende del método seleccionado.
        help_text="Valor entre 0 y 2 (exclusivo).", # Texto de ayuda.
        # Widget de entrada numérica con atributos para placeholder, pasos decimales y estilo.
        widget=forms.NumberInput(attrs={
            'placeholder': 'Ej: 1.25',
            'step': 'any', # Permite cualquier número decimal.
            'class': 'form-control' # Clase CSS para estilo.
        })
    )
    
    # Campo de selección para elegir el método a utilizar.
    method = forms.ChoiceField(
        label='Método', # Etiqueta visible.
        choices=METHOD_CHOICES_CH2, # Usa las opciones definidas arriba.
        widget=forms.Select(attrs={'class': 'form-select'}) # Widget de menú desplegable con estilo.
    )

    def clean(self):
        """
        Método de validación general del formulario. Se ejecuta después de
        la validación individual de cada campo. Es útil para validaciones
        que dependen de múltiples campos.

        Aquí, se usa para validar el campo 'omega_sor' condicionalmente,
        solo cuando se selecciona el método 'sor' o 'compare_all_ch2'.
        """
        # Llama al método clean() de la clase padre para obtener los datos limpios.
        cleaned_data = super().clean()
        # Obtiene los valores de omega y del método seleccionado.
        omega = cleaned_data.get('omega_sor')
        method = cleaned_data.get('method')

        # Comprueba si el método seleccionado requiere el valor de omega.
        if method and (method == 'sor' or method == 'compare_all_ch2'):
            # Si omega es requerido pero no se proporcionó, añade un error al campo 'omega_sor'.
            if omega is None:
                self.add_error('omega_sor', "El factor ω es requerido para el método SOR o para Comparar Todos.")
            # Si omega se proporcionó pero está fuera del rango válido (0 < omega < 2), añade un error.
            elif not (0 < omega < 2): 
                self.add_error('omega_sor', "El factor ω debe estar estrictamente entre 0 y 2.")

        # Siempre devuelve el diccionario de datos limpios al final.
        return cleaned_data