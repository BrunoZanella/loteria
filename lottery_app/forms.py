from django import forms
from django.contrib.auth.models import User
from .models import LotteryGame, LotteryTicket, Coupon
from django.core.exceptions import ValidationError

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



class CouponForm(forms.Form):
    code = forms.CharField(max_length=20, label='Código do Cupom')

    def clean_code(self):
        code = self.cleaned_data['code']
        try:
            coupon = Coupon.objects.get(code=code, is_active=True)
            if not coupon.is_valid():
                raise ValidationError('Este cupom não é mais válido.')
            return code
        except Coupon.DoesNotExist:
            raise ValidationError('Cupom inválido.')