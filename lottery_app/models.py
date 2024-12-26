from django.db import models
from django.contrib.auth.models import User

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

    # Função para atualizar os bilhetes associados
    def update_tickets_after_concurso_update(self):
        if self.sorteados:  # Certifique-se de que temos números sorteados
            # Obtém todos os bilhetes associados ao concurso
            tickets = LotteryTicket.objects.filter(game=self, concurso=self.concurso)
            for ticket in tickets:
                ticket.sorteados = self.sorteados  # Atualiza os números sorteados no bilhete
                ticket.save()

    def save(self, *args, **kwargs):
        is_new = self.pk is None  # Verifica se o jogo é novo
        concurso_changed = False

        if not is_new:
            original = LotteryGame.objects.get(pk=self.pk)
            if original.concurso != self.concurso:  # Verifica se o concurso foi alterado
                concurso_changed = True
        
        super().save(*args, **kwargs)

        # Se o concurso foi alterado, atualiza os bilhetes
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
