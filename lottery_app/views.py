from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm
from .models import LotteryGame, LotteryTicket
from .forms import LotteryPlayForm, UserUpdateForm
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json
from django.contrib.auth import login, logout

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Conta criada com sucesso! Agora você pode fazer login.')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def user_logout(request):
    logout(request)
    messages.success(request, 'Você saiu com sucesso!')
    return redirect('home')


@login_required
def profile(request):
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Seu perfil foi atualizado com sucesso!')
            return redirect('profile')
    else:
        form = UserUpdateForm(instance=request.user)
    return render(request, 'lottery_app/profile.html', {'form': form})

@login_required
def home(request):
    return render(request, 'lottery_app/home.html', {
        'form': LotteryPlayForm(),
    })

@login_required
def history(request):
    tickets = LotteryTicket.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'lottery_app/history.html', {'tickets': tickets})

@login_required
def delete_ticket(request, ticket_id):
    ticket = get_object_or_404(LotteryTicket, id=ticket_id, user=request.user)
    ticket.delete()
    messages.success(request, 'Jogo excluído com sucesso!')
    return redirect('history')

def game_info(request, game_id):
    game = get_object_or_404(LotteryGame, id=game_id)
    return JsonResponse({
        'total_numbers': game.total_numbers,
        'numbers_to_choose': game.numbers_to_choose
    })

@login_required
def generate_numbers(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        game = get_object_or_404(LotteryGame, id=data['game_id'])
        method = data['method']
        
        if method == 'manual':
            numbers = data.get('numbers', [])
        elif method == 'auto':
            numbers = random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose)
        else:  # AI method
            numbers = generate_ai_numbers(game)
        
        return JsonResponse({'numbers': sorted(numbers)})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def save_ticket(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        game = get_object_or_404(LotteryGame, id=data['game_id'])
        numbers = data['numbers']
        method = data['method']
        
        ticket = LotteryTicket.objects.create(
            user=request.user,
            game=game,
            numbers=sorted(numbers),
            generation_method=method
        )
        
        return JsonResponse({
            'message': 'Jogo salvo com sucesso!',
            'ticket_id': ticket.id
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def generate_ai_numbers(game):
    try:
        df = pd.read_csv(game.historical_data.path)
        X = df.iloc[:-1].values
        y = df.iloc[1:].values
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        last_numbers = df.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_numbers)[0]
        
        numbers = sorted([
            round(num) for num in prediction[:game.numbers_to_choose]
            if 1 <= round(num) <= game.total_numbers
        ])
        
        while len(numbers) < game.numbers_to_choose:
            num = random.randint(1, game.total_numbers)
            if num not in numbers:
                numbers.append(num)

        return sorted(numbers)
    except Exception:
        return random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose)