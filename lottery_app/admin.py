from django.contrib import admin
from .models import LotteryTicket, LotteryGame

@admin.register(LotteryGame)
class LotteryGameAdmin(admin.ModelAdmin):
    list_display = ('name', 'total_numbers', 'numbers_to_choose', 'concurso', 'sorteados')
    search_fields = ('name', 'concurso')
    list_filter = ('concurso',)

@admin.register(LotteryTicket)
class LotteryTicketAdmin(admin.ModelAdmin):
    list_display = ('user', 'game', 'created_at', 'concurso', 'generation_method')
    search_fields = ('user__username', 'game__name', 'concurso')
    list_filter = ('generation_method', 'created_at', 'concurso')
    readonly_fields = ('created_at',)
