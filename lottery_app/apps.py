from django.apps import AppConfig
import threading
import schedule
import time
from lottery_app.baixar_jogos import executar_script  # Ajuste o caminho conforme necessário

class LotteryAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'lottery_app'

    def ready(self):
        # Inicia um thread para agendar a execução
        def start_scheduler():
            agendamento = "13:36"
            schedule.every().day.at(agendamento).do(executar_script)
            print(f"Agendamento configurado para {agendamento}.")

            while True:
                schedule.run_pending()
                time.sleep(1)

        thread = threading.Thread(target=start_scheduler, daemon=True)
        thread.start()
