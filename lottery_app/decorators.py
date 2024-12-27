from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from .models import Subscription

def subscription_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        subscription = Subscription.objects.filter(user=request.user).first()
        if not subscription or not subscription.is_valid():
            messages.warning(request, 'Esta funcionalidade requer uma assinatura ativa.')
            return redirect('subscription_status')
        return view_func(request, *args, **kwargs)
    return wrapper