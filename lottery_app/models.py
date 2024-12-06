from django.db import models
from django.contrib.auth.models import User

class LotteryGame(models.Model):
    name = models.CharField(max_length=100)
    total_numbers = models.IntegerField()
    numbers_to_choose = models.IntegerField()
    historical_data = models.FileField(upload_to='historical_data/', null=True, blank=True)

    def __str__(self):
        return self.name

class LotteryTicket(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    game = models.ForeignKey(LotteryGame, on_delete=models.CASCADE)
    numbers = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
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