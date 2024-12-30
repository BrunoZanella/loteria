from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from lottery_app.utils.mercadopago import send_expiration_notification
from models import Subscription

class Command(BaseCommand):
    help = 'Check subscriptions and send expiration notifications'

    def handle(self, *args, **kwargs):
        # Get subscriptions expiring in 10 days
        expiring_date = timezone.now() + timedelta(days=10)
        subscriptions = Subscription.objects.filter(
            end_date__date=expiring_date.date(),
            is_active=True
        )

        for subscription in subscriptions:
            send_expiration_notification(subscription.user)
            self.stdout.write(
                self.style.SUCCESS(f'Sent notification to {subscription.user.email}')
            )