from django.apps import AppConfig
import threading
import schedule
import time
from lottery_app.baixar_jogos import executar_script  # Ajuste o caminho conforme necessário
import os
from zoneinfo import ZoneInfo
from datetime import datetime

class LotteryAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'lottery_app'

    def ready(self):
        # Configurar o fuso horário
        SAO_PAULO_TZ = ZoneInfo("America/Sao_Paulo")

        # Verifica se o processo atual é o principal
        if os.environ.get('RUN_MAIN') == 'true':
            def start_scheduler():
                # Definir o horário de agendamento
                agendamento = "15:15"
                print(f"Agendamento configurado para {agendamento} no fuso horário São Paulo ({SAO_PAULO_TZ}).")
                
                # Definir a tarefa com o fuso horário
                schedule.every().day.at(agendamento).do(executar_script)

                while True:
                    # Verificar e executar tarefas agendadas
                    now = datetime.now(tz=SAO_PAULO_TZ)
                #    print(f"Verificando tarefas agendadas às {now.strftime('%Y-%m-%d %H:%M:%S')} (horário de São Paulo).")
                    schedule.run_pending()
                    time.sleep(1)

            thread = threading.Thread(target=start_scheduler, daemon=True)
            thread.start()
