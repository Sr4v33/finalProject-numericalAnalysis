from django import forms

METHOD_CHOICES = [
    ('biseccion', 'Bisección'),
    ('regla_falsa', 'Regla Falsa'),
    ('punto_fijo', 'Punto Fijo'),
    ('newton', 'Newton'),
    ('secante', 'Secante'),
    ('newton_modificado', 'Newton Modificado'),
]

class NumericalMethodForm(forms.Form):
    function_f = forms.CharField(label='Función f(x)', required=False, 
                                widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ej: x**3 - cos(x)'}))
    function_g = forms.CharField(label='Función g(x)', required=False, 
                                widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ej: (cos(x))**(1/3)'}))
    method = forms.ChoiceField(label='Método', choices=METHOD_CHOICES, 
                               widget=forms.Select(attrs={'class': 'form-select'}))
    x0_xi = forms.FloatField(label='Xi / X0', required=False, 
                             widget=forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'}))
    xs_x1 = forms.FloatField(label='Xs / X1', required=False, 
                             widget=forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'}))
    tolerance = forms.FloatField(label='Tolerancia', initial=1e-5, 
                                 widget=forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'}))
    n_iterations = forms.IntegerField(label='Máx. Iteraciones', initial=100, 
                                      widget=forms.NumberInput(attrs={'class': 'form-control'}))

    def clean(self):
        cleaned_data = super().clean()
        method = cleaned_data.get('method')
        
        # Validaciones básicas (puedes añadir más)
        if not cleaned_data.get('function_f') and method != 'punto_fijo':
             raise forms.ValidationError("Se requiere f(x) para este método.")
        if method == 'punto_fijo' and not cleaned_data.get('function_g'):
             raise forms.ValidationError("Se requiere g(x) para Punto Fijo.")
        if method in ['biseccion', 'regla_falsa', 'newton', 'secante', 'newton_modificado', 'punto_fijo'] and cleaned_data.get('x0_xi') is None:
             raise forms.ValidationError("Se requiere Xi / X0.")
        if method in ['biseccion', 'regla_falsa', 'secante'] and cleaned_data.get('xs_x1') is None:
             raise forms.ValidationError("Se requiere Xs / X1.")
             
        return cleaned_data