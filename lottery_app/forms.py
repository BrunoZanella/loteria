from django import forms
from django.contrib.auth.models import User
from .models import LotteryGame, LotteryTicket

class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'

class LotteryPlayForm(forms.Form):
    game = forms.ModelChoiceField(
        queryset=LotteryGame.objects.all(),
        label='Jogo',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    number_of_tickets = forms.IntegerField(
        min_value=1,
        max_value=50,
        initial=1,
        label='Quantidade de Jogos',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    generation_method = forms.ChoiceField(
        choices=[
            ('manual', 'Manual'),
            ('auto', 'Automático'),
            ('ai', 'Inteligência Artificial')
        ],
        label='Método de Geração',
        widget=forms.Select(attrs={'class': 'form-select'})
    )