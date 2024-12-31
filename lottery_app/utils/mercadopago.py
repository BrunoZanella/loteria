import mercadopago
from django.conf import settings
from django.core.mail import send_mail
from django.urls import reverse
from django.template.loader import render_to_string
import uuid

def create_payment_preference(request):
    """Create Mercado Pago payment preference with payer info"""
    sdk = mercadopago.SDK(settings.MERCADOPAGO_ACCESS_TOKEN)
    
    # Generate unique ID for idempotency
    idempotency_key = str(uuid.uuid4())
    
    request_options = mercadopago.config.RequestOptions()
    request_options.custom_headers = {
        'x-idempotency-key': idempotency_key
    }
    
    # Get absolute URLs for callbacks
    base_url = request.build_absolute_uri('/')[:-1]

    # Generate external reference (e.g., user ID)
    external_reference = f"user_{request.user.id}_subscription_{uuid.uuid4()}"
    
    # Prepare payment data
    payment_data = {
        "items": [
            {
                "id": "subscription_monthly",
                "title": "Assinatura SortePlay",
                "description": "Assinatura mensal ao SortePlay",
                "category_id": "Assinatura",
                "quantity": 1,
                "currency_id": "BRL",
                "unit_price": 3.0,
            }
        ],
        "back_urls": {
            "success": f"{base_url}{reverse('payment_success')}",
            "failure": f"{base_url}{reverse('payment_failure')}",
            "pending": f"{base_url}{reverse('payment_pending')}"
        },
        "auto_return": "approved",
        "external_reference": external_reference,
        "payer": {
            "first_name": request.user.first_name,  # Nome do comprador
            "last_name": request.user.last_name,    # Sobrenome do comprador
            "email": request.user.email            # E-mail do comprador (opcional)
        }
    }
    
    try:
        # Create preference
        result = sdk.preference().create(payment_data, request_options)
        
        if result["status"] == 201:
            print(result["response"])
            return result["response"]
        else:
            print("Erro ao criar preferência:", result)
            return None
    except Exception as e:
        print("Erro ao chamar a API do Mercado Pago:", str(e))
        return None



def check_payment_status(payment_id):
    """Check payment status in Mercado Pago"""
    sdk = mercadopago.SDK(settings.MERCADOPAGO_ACCESS_TOKEN)
    payment_info = sdk.payment().get(payment_id)

    if payment_info["status"] == 200:
        external_reference = payment_info["response"].get("external_reference")
        
        # Use o external_reference para verificar no banco
        # Exemplo:
        # payment = Payment.objects.filter(external_reference=external_reference).first()
        # if payment:
        #     payment.status = payment_info["response"]["status"]
        #     payment.save()

        return payment_info["response"]

    return None


def send_expiration_notification(user):
    """Send subscription expiration notification email"""
    subject = 'Sua assinatura está próxima do vencimento'
    html_message = render_to_string('lottery_app/emails/subscription_expiring.html', {
        'user': user,
        'expiration_date': user.subscription.end_date.strftime('%d/%m/%Y')
    })
    
    send_mail(
        subject=subject,
        message='',
        html_message=html_message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
        fail_silently=False,
    )




