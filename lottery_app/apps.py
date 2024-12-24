




from django.apps import AppConfig
import threading
import schedule
import time
import requests
import os
from zoneinfo import ZoneInfo
from datetime import datetime, time as dt_time
from lottery_app.baixar_jogos import executar_script
from lottery_app.tasks import start_scheduler  # Importa o agendador
import unidecode  

import threading
import requests
import os
from django.apps import AppConfig

class LotteryAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'lottery_app'

    def ready(self):
        # Evitar loop em ambientes onde o servidor roda múltiplos workers
        if os.environ.get("RUN_MAIN", None) == "true":
            print("Chamando a view para iniciar tarefas...")
            try:
                # Substitua pela URL completa de produção
                requests.get("https://loteria.up.railway.app/start-tasks/")
            except Exception as e:
                print(f"Erro ao iniciar tarefas: {str(e)}")



'''
class LotteryAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'lottery_app'

    def ready(self):
        # Configurar o fuso horário
        SAO_PAULO_TZ = ZoneInfo("America/Sao_Paulo")

        # Verifica se o processo atual é o principal
        if os.environ.get('RUN_MAIN') == 'true':

            # Função para formatar os nomes dos jogos
            def format_game_name(game_name):
                """
                Formata o nome do jogo para corresponder ao formato esperado pela API.
                - Remove acentos
                - Remove espaços, traços
                - Converte para letras minúsculas
                """
                # Remove acentos, espaços e traços
                formatted_name = unidecode.unidecode(game_name).replace(" ", "").replace("-", "").lower()

                return formatted_name

            # Nova lógica para verificar mudanças de concurso
            def check_lottery_updates():
                from lottery_app.models import LotteryGame

                while True:
                    now = datetime.now(tz=SAO_PAULO_TZ)
                    current_time = now.time()
                    hora_api = dt_time(19, 0)
                    hora_fim_api = dt_time(23, 59)
                    
                    # Verificar se o horário está entre 16:00 e 23:59
                    if hora_api <= current_time < dt_time(23, 59):
                        print(f"Verificando atualizações de loteria às {now.strftime('%Y-%m-%d %H:%M:%S')}...")

                        # Iterar sobre todos os jogos registrados no banco de dados
                        for game in LotteryGame.objects.all():
                            try:
                                # Formata o nome do jogo
                                formatted_name = format_game_name(game.name)
                                api_url = f"https://loteriascaixa-api.herokuapp.com/api/{formatted_name}/latest"
                                
                                response = requests.get(api_url)
                                if response.status_code == 200:
                                    data = response.json()
                                    api_concurso = int(data.get('concurso', 0))
                                    api_dezenas = ','.join(data.get('dezenas', []))

                                    if api_concurso > game.concurso:
                                        print(f"Atualizando jogo {game.name}: novo concurso {api_concurso}.")
                                        game.concurso = api_concurso
                                        game.sorteados = api_dezenas
                                        game.save()
                                    else:
                                        print(f"Sem alterações para o jogo {game.name}.")
                                else:
                                    print(f"Erro ao acessar a API para {game.name} ({formatted_name}): {response.status_code}")
                            except Exception as e:
                                print(f"Erro ao verificar atualizações para {game.name}: {str(e)}")
                        
                        # Parar verificações ao atingir 00:00
                        if current_time >= hora_fim_api:
                            print(f"{hora_fim_api} alcançado, parando verificações.")
                            break
                    else:
                        print(f"Aguardando até as {hora_api} para iniciar verificações.")
                    
                    
                    time.sleep(600)  # Verifica a cada 30 segundos

                # Espera até as 16:00 do próximo dia para reiniciar
                while now.time() < hora_api:
                    now = datetime.now(tz=SAO_PAULO_TZ)
                    time.sleep(60)

            # Inicia a thread
            thread_lottery_updates = threading.Thread(target=check_lottery_updates, daemon=True)
            thread_lottery_updates.start()

'''



'''
from django.apps import AppConfig
import threading
import schedule
import time
from lottery_app.baixar_jogos import executar_script
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
'''