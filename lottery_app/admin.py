from django.contrib import admin
from .models import LotteryTicket, LotteryGame


admin.site.register(LotteryTicket)
admin.site.register(LotteryGame)

