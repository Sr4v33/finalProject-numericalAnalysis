# chapterThree/forms.py

from django import forms
from django.core.exceptions import ValidationError
from django.forms import formset_factory
from django.forms.formsets import BaseFormSet

# Define las opciones disponibles para los métodos de interpolación.
# Cada tupla contiene el valor interno y la etiqueta visible para el usuario.
METHOD_CHOICES = [
    ('vandermonde', 'Vandermonde'),
    ('newton', 'Newton (Diferencias Divididas)'),
    ('lagrange', 'Lagrange'),
    ('spline_linear', 'Spline Lineal'),
    ('spline_cubic', 'Spline Cúbico'),
    ('compare', 'Comparar Todos'), # Opción para comparar todos los métodos.
]

class PointForm(forms.Form):
    """
    Formulario para ingresar un único punto (X, Y).

    Este formulario se utiliza dentro de un FormSet para permitir al usuario
    ingresar múltiples puntos de datos para la interpolación.
    """
    # Campo para la coordenada X del punto.
    x = forms.FloatField(
        label='X', # Etiqueta visible del campo.
        widget=forms.NumberInput(attrs={ # Widget para la entrada numérica.
            'class': 'form-control form-control-sm point-input', # Clases CSS para estilo.
            'placeholder': 'Valor X', # Texto de ayuda en el campo.
            'step': 'any' # Permite ingresar números decimales.
        })
    )
    # Campo para la coordenada Y del punto.
    y = forms.FloatField(
        label='Y', # Etiqueta visible del campo.
        widget=forms.NumberInput(attrs={ # Widget para la entrada numérica.
            'class': 'form-control form-control-sm point-input', # Clases CSS para estilo.
            'placeholder': 'Valor Y', # Texto de ayuda en el campo.
            'step': 'any' # Permite ingresar números decimales.
        })
    )

class InterpolationForm(forms.Form):
    """
    Formulario principal para seleccionar el método de interpolación
    y, opcionalmente, ingresar un punto para evaluar el error.
    """
    # Campo para seleccionar el método de interpolación.
    method = forms.ChoiceField(
        label='Selecciona el Método:', # Etiqueta visible del campo.
        choices=METHOD_CHOICES, # Opciones definidas previamente.
        widget=forms.Select(attrs={ # Widget de selección desplegable.
            'class': 'form-select', # Clase CSS para estilo.
            'id': 'method-select' # ID para manipulación con JavaScript.
        }))

    # Campo opcional para ingresar un valor X conocido para evaluar el error.
    x_eval = forms.FloatField(
        label='X para Evaluar Error', # Etiqueta visible.
        required=False, # No es obligatorio llenar este campo.
        widget=forms.NumberInput(attrs={
            'class': 'form-control form-control-sm', # Clases CSS.
            'placeholder': 'Valor X conocido', # Texto de ayuda.
            'step': 'any' # Permite decimales.
        })
    )
    # Campo opcional para ingresar el valor Y (F(x)) conocido correspondiente a x_eval.
    y_eval = forms.FloatField(
        label='F(x) para Evaluar Error', # Etiqueta visible.
        required=False, # No es obligatorio.
        widget=forms.NumberInput(attrs={
            'class': 'form-control form-control-sm', # Clases CSS.
            'placeholder': 'Valor Y conocido', # Texto de ayuda.
            'step': 'any' # Permite decimales.
        })
    )

class BasePointFormSet(BaseFormSet):
    """
    Clase base personalizada para el FormSet de puntos.
    Permite añadir validaciones a nivel del conjunto de formularios.
    """
    def clean(self):
        """
        Realiza validaciones en todo el conjunto de formularios.
        - Verifica que haya al menos 2 puntos.
        - Verifica que los valores de X no estén duplicados.
        """
        # Si ya hay errores a nivel de campo individual, no continúa.
        if any(self.errors):
            return

        # Obtiene los datos limpios de todos los formularios.
        cleaned_data = self.cleaned_data
        
        # Filtra los formularios, excluyendo aquellos marcados para eliminar.
        active_forms_data = [
            data for data in cleaned_data if data and not data.get('DELETE')
        ]

        # Validación: Se requieren al menos 2 puntos.
        if len(active_forms_data) < 2:
            raise ValidationError("Se requieren al menos 2 puntos válidos.")

        # Extrae los valores X de los formularios activos.
        x_values = [data['x'] for data in active_forms_data if 'x' in data]

        # Validación: Los valores de X no deben estar duplicados.
        # Compara la longitud de la lista original con la longitud de un conjunto
        # (que automáticamente elimina duplicados).
        if len(x_values) != len(set(x_values)):
            raise ValidationError("Los valores de X no deben estar duplicados.")
            
        # Aquí se podrían añadir más validaciones personalizadas si fuera necesario.

# Crea un FormSet utilizando la función formset_factory.
# Un FormSet es una colección de formularios (en este caso, PointForm).
PointFormSet = formset_factory(
    PointForm, # El formulario base para cada elemento del FormSet.
    formset=BasePointFormSet, # Usa nuestra clase base personalizada para validaciones.
    min_num=2, # Número mínimo de formularios (puntos) requeridos.
    extra=0, # No muestra formularios vacíos adicionales por defecto.
    max_num=8, # Número máximo de formularios permitidos.
    can_delete=True, # Permite al usuario marcar formularios para eliminarlos.
)