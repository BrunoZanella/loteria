# Generated by Django 5.0.1 on 2024-12-27 13:46

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('lottery_app', '0005_coupon_subscription_couponuse'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='coupon',
            options={'verbose_name': 'Cupom', 'verbose_name_plural': 'Cupons'},
        ),
        migrations.AlterModelOptions(
            name='couponuse',
            options={'verbose_name': 'Usuário com Cupom', 'verbose_name_plural': 'Usuários com Cupons'},
        ),
        migrations.AlterModelOptions(
            name='subscription',
            options={'verbose_name': 'Assinatura', 'verbose_name_plural': 'Assinaturas'},
        ),
        migrations.AlterField(
            model_name='coupon',
            name='code',
            field=models.CharField(max_length=20, unique=True, verbose_name='Código'),
        ),
        migrations.AlterField(
            model_name='coupon',
            name='created_by',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL, verbose_name='Criado por'),
        ),
        migrations.AlterField(
            model_name='coupon',
            name='current_uses',
            field=models.IntegerField(default=0, verbose_name='Utilizações atuais'),
        ),
        migrations.AlterField(
            model_name='coupon',
            name='discount_type',
            field=models.CharField(choices=[('percentage', 'Porcentagem'), ('fixed', 'Valor Fixo')], max_length=10, verbose_name='Tipo de desconto'),
        ),
        migrations.AlterField(
            model_name='coupon',
            name='discount_value',
            field=models.DecimalField(decimal_places=2, max_digits=5, verbose_name='Valor do desconto'),
        ),
        migrations.AlterField(
            model_name='coupon',
            name='is_active',
            field=models.BooleanField(default=True, verbose_name='Está ativo'),
        ),
        migrations.AlterField(
            model_name='coupon',
            name='max_uses',
            field=models.IntegerField(default=1, verbose_name='Máximo de utilizações'),
        ),
        migrations.AlterField(
            model_name='coupon',
            name='valid_from',
            field=models.DateTimeField(verbose_name='Válido a partir de'),
        ),
        migrations.AlterField(
            model_name='coupon',
            name='valid_until',
            field=models.DateTimeField(verbose_name='Válido até'),
        ),
        migrations.AlterField(
            model_name='couponuse',
            name='coupon',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='lottery_app.coupon', verbose_name='Cupom'),
        ),
        migrations.AlterField(
            model_name='couponuse',
            name='subscription',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='lottery_app.subscription', verbose_name='Cupom'),
        ),
        migrations.AlterField(
            model_name='couponuse',
            name='used_at',
            field=models.DateTimeField(auto_now_add=True, verbose_name='Utilizado em'),
        ),
        migrations.AlterField(
            model_name='couponuse',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, verbose_name='Usuário'),
        ),
        migrations.AlterField(
            model_name='subscription',
            name='auto_renew',
            field=models.BooleanField(default=False, verbose_name='Renovação automática'),
        ),
        migrations.AlterField(
            model_name='subscription',
            name='end_date',
            field=models.DateTimeField(blank=True, null=True, verbose_name='Data de término'),
        ),
        migrations.AlterField(
            model_name='subscription',
            name='is_active',
            field=models.BooleanField(default=False, verbose_name='Está ativa'),
        ),
        migrations.AlterField(
            model_name='subscription',
            name='last_payment_date',
            field=models.DateTimeField(blank=True, null=True, verbose_name='Data do último pagamento'),
        ),
        migrations.AlterField(
            model_name='subscription',
            name='start_date',
            field=models.DateTimeField(blank=True, null=True, verbose_name='Data de início'),
        ),
        migrations.AlterField(
            model_name='subscription',
            name='user',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, verbose_name='Usuário'),
        ),
    ]