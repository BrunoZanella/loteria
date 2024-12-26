import schedule
import time
import threading
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
import unidecode  # Certifique-se de instalar: pip install unidecode
from lottery_app.baixar_jogos import executar_script

SAO_PAULO_TZ = ZoneInfo("America/Sao_Paulo")

# Defina as variáveis de horário de início e fim aqui
START_TIME_HOUR = 21
END_TIME_HOUR = 23
END_TIME_MINUTE = 59

def format_game_name(game_name):
    """
    Formata o nome do jogo para corresponder ao formato esperado pela API.
    - Remove acentos
    - Remove espaços, traços
    - Converte para letras minúsculas
    """
    return unidecode.unidecode(game_name).replace(" ", "").replace("-", "").lower()


def get_current_concurso(game_name):
    """
    Função para pegar o concurso atual de um jogo do banco de dados.
    """
    from lottery_app.models import LotteryGame
    game = LotteryGame.objects.filter(name=game_name).first()
    if game:
        return game.concurso
    return 0


def check_lottery_updates():
    """
    Função principal para verificar e atualizar os concursos da loteria.
    """
    from lottery_app.models import LotteryGame

    now = datetime.now(tz=SAO_PAULO_TZ)
    start_time = now.replace(hour=START_TIME_HOUR, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=END_TIME_HOUR, minute=END_TIME_MINUTE, second=0, microsecond=0)

    if now >= end_time:
        print("Fora do intervalo de verificação (10:00 - 11:00).")
        return

    print(f"Iniciando verificação às {now.strftime('%Y-%m-%d %H:%M:%S')}...")

    games_to_check = list(LotteryGame.objects.all())  # Carrega todos os jogos
    games_verified = []  # Lista de jogos verificados (concursos já atualizados)
    while games_to_check and now < end_time:
        remaining_games = []

        for game in games_to_check:
            try:
                # Primeiro, recupere o concurso do banco de dados
                db_concurso = get_current_concurso(game.name)

                # Se o concurso já foi verificado, pule o jogo
                if game.name in games_verified:
                    continue

                # Consulte a API para obter o concurso mais recente
                formatted_name = format_game_name(game.name)
                api_url = f"https://loteriascaixa-api.herokuapp.com/api/{formatted_name}/latest"
                response = requests.get(api_url)

                if response.status_code == 200:

                    data = response.json()
                    api_concurso = int(data.get('concurso', 0))
                    api_dezenas = ','.join(data.get('dezenas', []))

                    # Se o concurso da API for maior, atualize o banco de dados
                    if api_concurso > db_concurso:
                        print(f"Atualizando jogo {game.name}: novo concurso {api_concurso}.")
                        game.concurso = api_concurso
                        game.sorteados = api_dezenas
                        game.save()
                        games_verified.append(game.name)  # Marque como verificado
                    elif api_concurso == db_concurso:
                        print(f"Concurso do jogo {game.name} já está atualizado. Aguardando 5 minutos para nova tentativa.")
                        remaining_games.append(game)  # Reagendar a verificação após 5 minutos
                else:
                    print(f"Erro ao acessar a API para {game.name}: {response.status_code}")
                    remaining_games.append(game)  # Tentar novamente mais tarde
            except Exception as e:
                print(f"Erro ao verificar atualizações para {game.name}: {str(e)}")
                remaining_games.append(game)  # Reagendar para evitar perda

        # Atualiza a lista de jogos a verificar apenas com os jogos restantes
        games_to_check = remaining_games
        if games_to_check:
            print(f"Rechecando em 5 minutos... ({len(games_to_check)} jogos restantes)")
            time.sleep(300)  # Aguarda 5 minutos antes de rechecagem
        now = datetime.now(tz=SAO_PAULO_TZ)

    print("Fim do período de verificação. Próxima execução será às 10:00 do próximo dia.")

    # Limpeza da lista de verificados após as 11:00
    if now >= end_time:
        print("Limpando lista de jogos verificados para o próximo ciclo.")
        games_verified.clear()  # Limpa a lista para o próximo ciclo
        executar_script()

def start_scheduler():
    """
    Configura e inicia o agendador.
    """
    schedule.every().day.at(f"{START_TIME_HOUR:02d}:00").do(check_lottery_updates)
    print("Agendador iniciado. Aguardando tarefas...")

    while True:
        # Verifica se já passou das 10:00 mas ainda está antes das 11:00
        now = datetime.now(tz=SAO_PAULO_TZ)
        if now.hour >= START_TIME_HOUR and now < now.replace(hour=END_TIME_HOUR, minute=END_TIME_MINUTE, second=0, microsecond=0):
            check_lottery_updates()
        schedule.run_pending()
        time.sleep(1)  # Para evitar uso excessivo de CPU


def start_background_scheduler():
    """
    Inicia o agendador em uma thread separada.
    """
    
    thread = threading.Thread(target=start_scheduler, daemon=True)
    thread.start()
    print("Thread do agendador iniciada.")
