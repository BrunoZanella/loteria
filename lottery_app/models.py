from django.db import models
from django.contrib.auth.models import User

class LotteryGame(models.Model):
    name = models.CharField(max_length=100)
    total_numbers = models.IntegerField()
    numbers_to_choose = models.IntegerField()
    historical_data = models.FileField(upload_to='historical_data/', null=True, blank=True)
    concurso = models.IntegerField(default=0)
    sorteados = models.CharField(max_length=200, blank=True, null=True)

    def __str__(self):
        return self.name

    def get_sorted_numbers(self):
        if not self.sorteados:
            return []
        return [int(num.strip()) for num in self.sorteados.split(',')]

class LotteryTicket(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    game = models.ForeignKey(LotteryGame, on_delete=models.CASCADE)
    numbers = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    concurso = models.IntegerField()
    generation_method = models.CharField(
        max_length=20,
        choices=[
            ('manual', 'Manual'),
            ('auto', 'Automático'),
            ('ai', 'Inteligência Artificial')
        ]
    )

    def __str__(self):
        return f"{self.game.name} - {self.created_at.strftime('%d/%m/%Y %H:%M')}"

    def check_matches(self):
        if not self.game.sorteados:
            return [], []
        
        winning_numbers = self.game.get_sorted_numbers()
        matches = []
        non_matches = []
        
        for number in self.numbers:
            if number in winning_numbers:
                matches.append(number)
            else:
                non_matches.append(number)
                
        return matches, non_matches

    def is_winner(self):
        if not self.game.sorteados:
            return False
        matches, _ = self.check_matches()
        return len(matches) == self.game.numbers_to_choose