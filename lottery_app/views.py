from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm
from .models import LotteryGame, LotteryTicket
from .forms import LotteryPlayForm, UserUpdateForm
from .utils.number_analysis import generate_ai_numbers
import random
import json

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
        'numbers_to_choose': game.numbers_to_choose,
        'concurso': game.concurso
    })

@login_required
def generate_numbers(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        game = get_object_or_404(LotteryGame, id=data['game_id'])
        method = data['method']
        num_tickets = data.get('number_of_tickets', 1)
        
        if method == 'manual':
            numbers = data.get('numbers', [])
            return JsonResponse({'numbers': [numbers]})  # Return as list for consistency
        elif method == 'auto':
            # Generate unique random combinations
            predictions = []
            while len(predictions) < num_tickets:
                numbers = random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose)
                if sorted(numbers) not in predictions:
                    predictions.append(sorted(numbers))
            return JsonResponse({'numbers': predictions})
        else:  # AI method
            predictions = generate_ai_numbers(game, num_tickets)
            if not isinstance(predictions, list):
                predictions = [predictions]
            return JsonResponse({'numbers': predictions})
    
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
            generation_method=method,
            concurso=game.concurso,
            sorteados=game.sorteados
        )
        
        return JsonResponse({
            'message': 'Jogo salvo com sucesso!',
            'ticket_id': ticket.id
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)



from django.http import JsonResponse
import threading
from lottery_app.tasks import start_scheduler

def start_background_tasks(request):
    """
    Inicia as tarefas em segundo plano.
    """
    if not hasattr(threading.current_thread(), "_scheduler_thread"):
        print("Iniciando agendador...")
        thread_scheduler = threading.Thread(target=start_scheduler, daemon=True)
        threading.current_thread()._scheduler_thread = thread_scheduler
        thread_scheduler.start()

    return JsonResponse({"status": "Tarefas em segundo plano iniciadas"})
