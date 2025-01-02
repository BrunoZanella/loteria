from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm
from .models import LotteryGame, LotteryTicket, Subscription, Coupon, CouponUse
from .forms import LotteryPlayForm, UserUpdateForm, CouponForm
from .utils.number_analysis import generate_ai_numbers
import random
import json
from django.core.cache import cache
from datetime import datetime
from django.contrib.auth import login, logout
import time
from decimal import Decimal
from .decorators import subscription_required
from django.urls import reverse
from lottery_app.utils.mercadopago import create_payment_preference, check_payment_status

@login_required
def create_payment(request):
    preference = create_payment_preference(request)
    return redirect(preference['init_point'])

#@login_required
def payment_success(request):
    payment_id = request.GET.get('payment_id')
    if payment_id:
        payment_info = check_payment_status(payment_id)
        if payment_info and payment_info['status'] == 'approved':
            external_reference = payment_info.get("external_reference")
            
            # Use o external_reference para confirmar e ativar a assinatura
            # Exemplo:
            # subscription = Subscription.objects.filter(user=request.user).first()
            # if subscription and external_reference:
            #     subscription.activate(months=1)

            messages.success(request, 'Pagamento aprovado! Sua assinatura está ativa.')
            return render(request, 'lottery_app/payment/success.html')
    
    messages.error(request, 'Não foi possível confirmar o pagamento.')
    return redirect('subscription_status')




#@login_required
def payment_failure(request):
    messages.error(request, 'O pagamento não foi aprovado. Por favor, tente novamente.')
    return render(request, 'lottery_app/payment/failure.html')

#@login_required
def payment_pending(request):
    messages.warning(request, 'Seu pagamento está pendente de aprovação.')
    return render(request, 'lottery_app/payment/pending.html')













def user_logout(request):
    logout(request)
    messages.success(request, 'Você saiu com sucesso!')
    return redirect('home')

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

# @login_required
# def profile(request):
#     if request.method == 'POST':
#         form = UserUpdateForm(request.POST, instance=request.user)
#         if form.is_valid():
#             form.save()
#             messages.success(request, 'Seu perfil foi atualizado com sucesso!')
#             return redirect('profile')
#     else:
#         form = UserUpdateForm(instance=request.user)
#     return render(request, 'lottery_app/profile.html', {'form': form})

@login_required
def profile(request):
    # Obtém a assinatura do usuário
    subscription = Subscription.objects.filter(user=request.user).first()

    # Se a assinatura não for válida, exibe a mensagem de erro, mas mantém o formulário
    is_active = subscription.is_valid() if subscription else False

    # Se o usuário não tiver uma assinatura ativa, o formulário de perfil também será renderizado
    form = UserUpdateForm(instance=request.user)

    context = {
        'subscription': subscription,
        'is_active': is_active,
        'coupon_form': CouponForm(),
        'form': form
    }

    return render(request, 'lottery_app/profile.html', context)



def home(request):
    """
    Exibe a página principal e inicia as tarefas em segundo plano, se necessário.
    """
    # Verifica se o agendador já foi iniciado hoje
    today = datetime.now().date()
    last_run_date = cache.get("last_scheduler_run_date")

    # Se o agendador não foi executado hoje, chama a função para iniciar as tarefas em segundo plano
    if last_run_date != today:
        start_background_tasks(request)  # Chama a função para iniciar o agendador
        # Armazena a data da última execução no cache
        cache.set("last_scheduler_run_date", today, timeout=86400)  # Expira após 24 horas

    # Verifica se o usuário está autenticado
    has_subscription = False
    has_subscription = True # por enquanto todo mundo liberado
    if request.user.is_authenticated:
        # Verifica o status da assinatura
        subscription = Subscription.objects.filter(user=request.user).first()
        has_subscription = subscription.is_valid() if subscription else False

    # apagar essa linha para verificar a autorizacao
    has_subscription = True # por enquanto todo mundo liberado
    
    return render(request, 'lottery_app/home.html', {
        'form': LotteryPlayForm(),
        'has_subscription': has_subscription,
        'is_authenticated': request.user.is_authenticated  # Passa o status de autenticação
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
            time.sleep(3)  # Atraso de 1 segundo para simular um tempo maior de processamento
            predictions = []
            while len(predictions) < num_tickets:
                numbers = random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose)
                if numbers not in predictions:  # Não precisa de sorted(), porque números já estão aleatórios
                    predictions.extend(numbers)  # Use extend para adicionar os números diretamente
            
            return JsonResponse({'numbers': predictions})
        else:  # AI method
            time.sleep(3)  # Atraso de 1 segundo para simular um tempo maior de processamento
            predictions = generate_ai_numbers(game, num_tickets)
            if not isinstance(predictions, list):
                predictions = [predictions]
            return JsonResponse({'numbers': predictions})

    return JsonResponse({'error': 'Invalid request'}, status=400)

        # else:  # AI method
        #     predictions = generate_ai_numbers(game, num_tickets)
        #     if not isinstance(predictions, list):
        #         predictions = [predictions]
        #     return JsonResponse({'numbers': predictions})
        

@login_required
def save_ticket(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        game = get_object_or_404(LotteryGame, id=data['game_id'])
        numbers = data['numbers']
        method = data['method']

        # Incrementa o número do concurso para o próximo
        next_concurso = game.concurso + 1

        # Salva o bilhete com o próximo número do concurso
        ticket = LotteryTicket.objects.create(
            user=request.user,
            game=game,
            numbers=sorted(numbers),
            generation_method=method,
            concurso=next_concurso,  # Define como o próximo concurso
            sorteados=None  # Os números sorteados ainda não existem para o próximo concurso
        )

        return JsonResponse({
            'message': 'Jogo salvo com sucesso para o próximo concurso!',
            'ticket_id': ticket.id
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)


# @login_required
# def save_ticket(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         game = get_object_or_404(LotteryGame, id=data['game_id'])
#         numbers = data['numbers']
#         method = data['method']
        
#         ticket = LotteryTicket.objects.create(
#             user=request.user,
#             game=game,
#             numbers=sorted(numbers),
#             generation_method=method,
#             concurso=game.concurso,
#             sorteados=game.sorteados
#         )
        
#         return JsonResponse({
#             'message': 'Jogo salvo com sucesso!',
#             'ticket_id': ticket.id
#         })
    
#     return JsonResponse({'error': 'Invalid request'}, status=400)





from django.http import JsonResponse
import threading
from lottery_app.tasks import start_scheduler

def start_background_tasks(request):
    """
    Inicia as tarefas em segundo plano.
    """
    if not hasattr(threading.current_thread(), "_scheduler_thread"):
        print("\nIniciando agendador...")
        thread_scheduler = threading.Thread(target=start_scheduler, daemon=True)
        threading.current_thread()._scheduler_thread = thread_scheduler
        thread_scheduler.start()

    return JsonResponse({"status": "Tarefas em segundo plano iniciadas"})




@login_required
def subscription_status(request):
    subscription = Subscription.objects.filter(user=request.user).first()
    context = {
        'subscription': subscription,
        'is_active': subscription.is_valid() if subscription else False,
        'coupon_form': CouponForm()
    }
    return render(request, 'lottery_app/profile.html', context)

@login_required
def apply_coupon(request):
    if request.method == 'POST':
        form = CouponForm(request.POST)
        if form.is_valid():
            code = form.cleaned_data['code']
            coupon = Coupon.objects.get(code=code)

            # Valor mensal padrão
            monthly_price = Decimal('3.00')

            # Calcula desconto
            discount = coupon.calculate_discount(monthly_price)
            final_price = monthly_price - discount

            # Se o cupom for de 100% ou o preço final for 0
            if final_price <= Decimal('0.00'):
                subscription, created = Subscription.objects.get_or_create(user=request.user)
                subscription.activate(months=1)
                
                # Registra uso do cupom
                CouponUse.objects.create(
                    coupon=coupon,
                    user=request.user,
                    subscription=subscription
                )

                coupon.use()
                messages.success(request, 'Cupom aplicado com sucesso! Sua assinatura está ativa.')
            else:
                # Aqui você pode redirecionar para a página de pagamento
                # com o preço com desconto
                messages.info(request, f'Valor com desconto: R$ {final_price:.2f}')
            
            return redirect('subscription_status')
    
    messages.error(request, 'Cupom inválido.')
    return redirect('subscription_status')