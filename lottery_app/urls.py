from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('generate/', views.generate_numbers, name='generate_numbers'),
    path('save-ticket/', views.save_ticket, name='save_ticket'),
    path('history/', views.history, name='history'),
    path('delete-ticket/<int:ticket_id>/', views.delete_ticket, name='delete_ticket'),
    path('profile/', views.profile, name='profile'),
    path('register/', views.register, name='register'),
    path('api/game/<int:game_id>/', views.game_info, name='game_info'),
    path('start_tasks/', views.start_background_tasks, name='start_tasks'),
]