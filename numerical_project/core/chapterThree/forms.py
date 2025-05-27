# chapterThree/forms.py

from django import forms
from django.core.exceptions import ValidationError
from django.forms import formset_factory
from django.forms.formsets import BaseFormSet

METHOD_CHOICES = [
    ('vandermonde', 'Vandermonde'),
    ('newton', 'Newton (Diferencias Divididas)'),
    ('lagrange', 'Lagrange'),
    ('spline_linear', 'Spline Lineal'),
    ('spline_cubic', 'Spline Cúbico'),
]

class PointForm(forms.Form):
    """Formulario para un único punto (X, Y)."""
    x = forms.FloatField(
        label='X',
        widget=forms.NumberInput(attrs={
            'class': 'form-control form-control-sm point-input',
            'placeholder': 'Valor X',
            'step': 'any' # Permite decimales
        })
    )
    y = forms.FloatField(
        label='Y',
        widget=forms.NumberInput(attrs={
            'class': 'form-control form-control-sm point-input',
            'placeholder': 'Valor Y',
            'step': 'any'
        })
    )

class InterpolationForm(forms.Form):
    """Formulario principal para seleccionar el método."""
    method = forms.ChoiceField(
        label='Selecciona el Método:',
        choices=METHOD_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )


# Puedes añadir validaciones personalizadas al FormSet si es necesario
class BasePointFormSet(BaseFormSet):
    def clean(self):
        """Validación a nivel de FormSet (ej: verificar X duplicados)."""
        if any(self.errors):
            # No hacer validaciones adicionales si hay errores de campo
            return

        cleaned_data = self.cleaned_data
        
        # Filtrar formularios marcados para eliminar
        active_forms_data = [
            data for data in cleaned_data if data and not data.get('DELETE')
        ]

        if len(active_forms_data) < 2:
            raise ValidationError("Se requieren al menos 2 puntos válidos.")

        x_values = [data['x'] for data in active_forms_data if 'x' in data]

        if len(x_values) != len(set(x_values)):
            raise ValidationError("Los valores de X no deben estar duplicados.")
            
        # Puedes añadir más validaciones aquí

# Usaremos BasePointFormSet en la vista
PointFormSet = formset_factory(
    PointForm,
    formset=BasePointFormSet, # Usamos nuestra clase base con validación
    min_num=2,
    extra=0, 
    max_num=8,
    can_delete=True,

)