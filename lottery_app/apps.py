from django.apps import AppConfig
import threading
import schedule
import time
from lottery_app.baixar_jogos import executar_script  # Ajuste o caminho conforme necessário
import os

class LotteryAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'lottery_app'

    def ready(self):
        # Verifica se o processo atual é o principal
        if os.environ.get('RUN_MAIN') == 'true':
            def start_scheduler():
                agendamento = "21:00"
                schedule.every().day.at(agendamento).do(executar_script)
                print(f"Agendamento configurado para {agendamento}.")

                while True:
                    schedule.run_pending()
                    time.sleep(1)

            thread = threading.Thread(target=start_scheduler, daemon=True)
            thread.start()
