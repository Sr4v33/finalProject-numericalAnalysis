from django import forms

METHOD_CHOICES_CH2 = [
    ('jacobi', 'Jacobi'),
    ('gauss_seidel', 'Gauss-Seidel'),
    ('sor', 'SOR (Successive Over-Relaxation)'),
    ('compare_all_ch2', 'Comparar Todos los Métodos'),
]

class MatrixMethodForm(forms.Form):
    matrix_A_str = forms.CharField(
        label='Matriz A (Max 7x7)',
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': 'Ej: [[4,1,0],[1,5,1],[0,1,6]]'}),
        help_text="Ingresa como lista de listas Python. Ej: [[1,2],[3,4]]",
        required=False 
    )
    vector_b_str = forms.CharField(
        label='Vector b',
        widget=forms.TextInput(attrs={'placeholder': 'Ej: [7,8,9]'}),
        help_text="Ingresa como lista Python. Ej: [1,2]",
        required=False 
    )
    vector_x0_str = forms.CharField(
        label='Vector inicial x0',
        widget=forms.TextInput(attrs={'placeholder': 'Ej: [0,0,0]'}),
        help_text="Ingresa como lista Python. Ej: [0,0]",
        required=False 
    )
    tolerance = forms.FloatField(label='Tolerancia', initial=1e-7)
    max_iterations = forms.IntegerField(label='Máx. Iteraciones', initial=100)
    omega_sor = forms.FloatField(
        label='Factor de Relajación ω (para SOR y Comparar Todos)',
        required=False, # Mantenlo como False, la lógica en clean() lo manejará
        help_text="Valor entre 0 y 2 (exclusivo).",
        widget=forms.NumberInput(attrs={
            'placeholder': 'Ej: 1.25',
            'step': 'any',
            'class': 'form-control'
        })
    )
    method = forms.ChoiceField(
        label='Método',
        choices=METHOD_CHOICES_CH2,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    # Elimina el método clean_omega_sor(self) si lo tenías separado

    def clean(self):
        cleaned_data = super().clean()
        omega = cleaned_data.get('omega_sor')
        method = cleaned_data.get('method')

        # Solo validar omega si el método es SOR o Comparar Todos
        if method and (method == 'sor' or method == 'compare_all_ch2'):
            if omega is None:
                # self.add_error(<nombre_del_campo>, <mensaje_de_error>)
                self.add_error('omega_sor', "El factor ω es requerido para el método SOR o para Comparar Todos.")
            elif not (0 < omega < 2): # Solo chequear el rango si omega no es None
                self.add_error('omega_sor', "El factor ω debe estar estrictamente entre 0 y 2.")

        return cleaned_data