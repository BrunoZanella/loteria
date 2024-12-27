from django.contrib import admin
from .models import LotteryTicket, LotteryGame,Subscription, Coupon, CouponUse

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

@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ('user', 'is_active', 'start_date', 'end_date', 'last_payment_date')
    
@admin.register(Coupon)
class CouponAdmin(admin.ModelAdmin):
    list_display = ('code', 'discount_type', 'discount_value', 'valid_from', 'is_active')

@admin.register(CouponUse)
class CouponUseAdmin(admin.ModelAdmin):
    list_display = ('coupon', 'user', 'used_at', 'subscription')