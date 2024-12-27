from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo
from django.db import models
from django.contrib.auth.models import User

# Definindo o timezone
SAO_PAULO_TZ = ZoneInfo("America/Sao_Paulo")

class LotteryGame(models.Model):
    name = models.CharField(
        max_length=100, 
        verbose_name="Nome do jogo"
    )
    total_numbers = models.IntegerField(
        verbose_name="Total de números disponíveis"
    )
    numbers_to_choose = models.IntegerField(
        verbose_name="Quantidade de números a escolher"
    )
    historical_data = models.FileField(
        upload_to='historical_data/', 
        null=True, 
        blank=True, 
        verbose_name="Arquivo de dados históricos"
    )
    concurso = models.IntegerField(
        default=0, 
        verbose_name="Número do concurso"
    )
    sorteados = models.CharField(
        max_length=200, 
        blank=True, 
        null=True, 
        verbose_name="Números sorteados"
    )

    def __str__(self):
        return self.name

    def get_sorted_numbers(self):
        if not self.sorteados:
            return []
        return [int(num.strip()) for num in self.sorteados.split(',')]

    def update_tickets_after_concurso_update(self):
        if self.sorteados:
            tickets = LotteryTicket.objects.filter(game=self, concurso=self.concurso)
            for ticket in tickets:
                ticket.sorteados = self.sorteados
                ticket.save()

    def save(self, *args, **kwargs):
        is_new = self.pk is None
        concurso_changed = False

        if not is_new:
            original = LotteryGame.objects.get(pk=self.pk)
            if original.concurso != self.concurso:
                concurso_changed = True
        
        super().save(*args, **kwargs)

        if concurso_changed:
            self.update_tickets_after_concurso_update()

    class Meta:
        verbose_name = "Jogo de Loteria"
        verbose_name_plural = "Jogos de Loteria"

class LotteryTicket(models.Model):
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        verbose_name="Usuário"
    )
    game = models.ForeignKey(
        LotteryGame, 
        on_delete=models.CASCADE, 
        verbose_name="Jogo"
    )
    numbers = models.JSONField(
        verbose_name="Números escolhidos"
    )
    created_at = models.DateTimeField(
        auto_now_add=True, 
        verbose_name="Data de criação"
    )
    concurso = models.IntegerField(
        verbose_name="Número do concurso"
    )
    sorteados = models.CharField(
        max_length=200, 
        blank=True, 
        null=True, 
        verbose_name="Números sorteados"
    )
    generation_method = models.CharField(
        max_length=20,
        choices=[('manual', 'Manual'), ('auto', 'Automático'), ('ai', 'Inteligência Artificial')],
        verbose_name="Método de geração"
    )

    def __str__(self):
        return f"{self.game.name} - {self.created_at.strftime('%d/%m/%Y %H:%M')}"

    def check_matches(self):
        if not self.sorteados:
            return [], []

        winning_numbers = [int(num.strip()) for num in self.sorteados.split(',')]
        matches = []
        non_matches = []

        for number in self.numbers:
            if number in winning_numbers:
                matches.append(number)
            else:
                non_matches.append(number)

        return matches, non_matches

    def is_winner(self):
        if not self.sorteados:
            return False
        matches, _ = self.check_matches()
        return len(matches) == self.game.numbers_to_choose

    class Meta:
        verbose_name = "Bilhete de Loteria"
        verbose_name_plural = "Bilhetes de Loteria"

class Subscription(models.Model):
    user = models.OneToOneField(
        User, 
        on_delete=models.CASCADE, 
        verbose_name="Usuário"
    )
    is_active = models.BooleanField(
        default=False, 
        verbose_name="Está ativa"
    )
    start_date = models.DateTimeField(
        null=True, 
        blank=True, 
        verbose_name="Data de início"
    )
    end_date = models.DateTimeField(
        null=True, 
        blank=True, 
        verbose_name="Data de término"
    )
    auto_renew = models.BooleanField(
        default=False, 
        verbose_name="Renovação automática"
    )
    last_payment_date = models.DateTimeField(
        null=True, 
        blank=True, 
        verbose_name="Data do último pagamento"
    )

    def is_valid(self):
        if not self.is_active:
            return False
        now = datetime.now(tz=SAO_PAULO_TZ)
        return self.start_date <= now <= self.end_date if self.start_date and self.end_date else False

    def activate(self, months=1, start_date=None):
        self.is_active = True
        start_date = start_date or datetime.now(tz=SAO_PAULO_TZ)
        self.start_date = start_date
        self.end_date = self.start_date + timedelta(days=30 * months)
        self.save()

    class Meta:
        verbose_name = "Assinatura"
        verbose_name_plural = "Assinaturas"

class Coupon(models.Model):
    DISCOUNT_TYPES = [
        ('percentage', 'Porcentagem'),
        ('fixed', 'Valor Fixo'),
    ]

    code = models.CharField(
        max_length=20, 
        unique=True, 
        verbose_name="Código"
    )
    discount_type = models.CharField(
        max_length=10, 
        choices=DISCOUNT_TYPES, 
        verbose_name="Tipo de desconto"
    )
    discount_value = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        verbose_name="Valor do desconto"
    )
    valid_from = models.DateTimeField(
        verbose_name="Válido a partir de"
    )
    valid_until = models.DateTimeField(
        verbose_name="Válido até"
    )
    max_uses = models.IntegerField(
        default=1, 
        verbose_name="Máximo de utilizações"
    )
    current_uses = models.IntegerField(
        default=0, 
        verbose_name="Utilizações atuais"
    )
    created_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        verbose_name="Criado por"
    )
    is_active = models.BooleanField(
        default=True, 
        verbose_name="Está ativo"
    )

    def is_valid(self):
        now = datetime.now(tz=SAO_PAULO_TZ)
        return (
            self.is_active and
            self.current_uses < self.max_uses and
            self.valid_from <= now <= self.valid_until
        )

    def calculate_discount(self, original_price):
        if not self.is_valid():
            return Decimal('0.00')
        
        if self.discount_type == 'percentage':
            return (Decimal(str(original_price)) * self.discount_value) / Decimal('100.00')
        return min(self.discount_value, Decimal(str(original_price)))

    def use(self):
        if self.is_valid():
            self.current_uses += 1
            self.save()
            return True
        return False

    class Meta:
        verbose_name = "Cupom"
        verbose_name_plural = "Cupons"

class CouponUse(models.Model):
    coupon = models.ForeignKey(
        Coupon, 
        on_delete=models.CASCADE, 
        verbose_name="Cupom"
    )
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        verbose_name="Usuário"
    )
    used_at = models.DateTimeField(
        auto_now_add=True, 
        verbose_name="Utilizado em"
    )
    subscription = models.ForeignKey(
        Subscription, 
        on_delete=models.CASCADE, 
       verbose_name="Cupom"
    )

    class Meta:
        verbose_name = "Usuário com Cupom"
        verbose_name_plural = "Usuários com Cupons"

