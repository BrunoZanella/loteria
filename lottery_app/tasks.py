import schedule
import time
import threading
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
import unidecode  # Certifique-se de instalar: pip install unidecode
from lottery_app.baixar_jogos import executar_script
import pandas as pd
import os

SAO_PAULO_TZ = ZoneInfo("America/Sao_Paulo")

# Defina as variáveis de horário de início e fim aqui
START_TIME_HOUR = 20
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
    from lottery_app.models import LotteryGame, LotteryTicket

    now = datetime.now(tz=SAO_PAULO_TZ)
    start_time = now.replace(hour=START_TIME_HOUR, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=END_TIME_HOUR, minute=END_TIME_MINUTE, second=0, microsecond=0)

    if now >= end_time:
        print("Fora do intervalo de verificação (10:00 - 11:00).")
        return

    print(f"Iniciando verificação às {now.strftime('%Y-%m-%d %H:%M:%S')}...")

    games_to_check = LotteryGame.objects.all()  # Carrega todos os jogos

    for game in games_to_check:
        now = datetime.now(tz=SAO_PAULO_TZ)
        if now >= end_time:
            print("Fim do período de verificação. Encerrando execução.")
            executar_script()
            return

        print(f"Verificando jogo: {game.name}")
        tickets_to_check = (
            LotteryTicket.objects.filter(game=game, sorteados__isnull=True)
            .values_list("concurso", flat=True)
            .distinct()
            .order_by("concurso")
        )

        if not tickets_to_check:
            print(f"Nenhum concurso pendente para o jogo {game.name}.")
            continue

        for concurso in tickets_to_check:
            now = datetime.now(tz=SAO_PAULO_TZ)
            if now >= end_time:
                print("Fim do período de verificação. Encerrando execução.")
                return

            formatted_name = format_game_name(game.name).lower()
            api_url = f"https://servicebus2.caixa.gov.br/portaldeloterias/api/{formatted_name}/{concurso}"
            print(f"Consultando API: {api_url}")

            try:
                response = requests.get(api_url)

                if response.status_code == 200:
                    data = response.json()
                    api_concurso = int(data.get("numero", 0))
                    api_dezenas = ",".join(data.get("listaDezenas", []))

                    if api_concurso == concurso:
                        print(f"Atualizando jogo {game.name} para o concurso {api_concurso}.")
                        game.concurso = api_concurso
                        game.sorteados = api_dezenas
                        game.save()

                        tickets = LotteryTicket.objects.filter(game=game, concurso=concurso)
                        for ticket in tickets:
                            ticket.sorteados = api_dezenas
                            ticket.save()
                    else:
                        print(f"Concurso {concurso} para o jogo {game.name} ainda não disponível.")
                elif response.status_code == 404:
                    print(f"Concurso {concurso} para o jogo {game.name} não encontrado. Tentando novamente mais tarde.")
                elif response.status_code == 500:
                    print(f"Concurso {concurso} para o jogo {game.name}. Ocorreu um erro inesperado.")
                else:
                    print(f"Erro na API para o jogo {game.name}, concurso {concurso}: {response.status_code}")
            except Exception as e:
                print(f"Erro ao verificar o jogo {game.name}, concurso {concurso}: {str(e)}")

            # Pausa de 30 segundos entre requisições
            time.sleep(30)

    print("Todos os concursos pendentes foram verificados ou atingiu o horário limite.")
#    executar_script()

    # Limpeza da lista de verificados após as 11:00
    if now >= end_time:
        print("Limpando lista de jogos verificados para o próximo ciclo.")
        executar_script()





'''
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

                # Incrementa o concurso para buscar o próximo
                next_concurso = db_concurso + 1

                # Consulte a nova API para obter o próximo concurso
                formatted_name = format_game_name(game.name).lower()
                api_url = f"https://servicebus2.caixa.gov.br/portaldeloterias/api/{formatted_name}/{next_concurso}"
                response = requests.get(api_url)

                if response.status_code == 200:
                    data = response.json()
                    api_concurso = int(data.get('numero', 0))
                    api_dezenas = ','.join(data.get('listaDezenas', []))

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
                elif response.status_code == 404:
                    # Concurso ainda não disponível
                    print(f"Concurso {next_concurso} para o jogo {game.name} ainda não foi lançado. Tentando novamente mais tarde.")
                    remaining_games.append(game)
                else:
                    # Outros erros
                    error_message = response.text
                    if "Ocorreu um erro inesperado" in error_message:
                        print(f"O concurso {next_concurso} para o jogo {game.name} ainda não saiu. Verifique novamente mais tarde.")
                    else:
                        print(f"Erro ao acessar a API para {game.name} (Concurso {next_concurso}): {response.status_code}")
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
'''






import joblib
import os

def train_all_models():
    """
    Treina modelos para todos os jogos utilizando os arquivos CSV disponíveis.
    """
    from lottery_app.models import LotteryGame
    from lottery_app.utils.number_analysis import (
        prepare_enhanced_lottery_data,
        create_advanced_features,
        train_ensemble_models
    )

    try:
        # Pasta onde os modelos serão salvos
        models_dir = "trained_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Pasta {models_dir} criada para salvar os modelos.")

        games = LotteryGame.objects.all()
        for game in games:
            print(f"Treinando modelo para o jogo: {game.name}")

            # Verificar se os dados históricos existem
            if not game.historical_data or not os.path.exists(game.historical_data.path):
                print(f"Dados históricos não encontrados para o jogo {game.name}.")
                continue

            # Carregar dados históricos
            df = pd.read_csv(game.historical_data.path)
            numbers_df = prepare_enhanced_lottery_data(df)

            # Criar features avançadas
            X, y = create_advanced_features(numbers_df)

            # Treinar ensemble de modelos
            models = train_ensemble_models(X, y)

            # Salvar os modelos treinados
            model_path = os.path.join(models_dir, f"{game.name.lower()}_rf_model.pkl")
            scaler_path = os.path.join(models_dir, f"{game.name.lower()}_scaler.pkl")
            
            if os.path.exists(model_path):
                print(f"Modelo antigo encontrado para {game.name}. Substituindo pelo novo.")
            
            joblib.dump(models['rf'], model_path)  # Salvar o modelo RandomForest
            joblib.dump(models['scaler'], scaler_path)  # Salvar o escalador
            print(f"Novo modelo salvo em {model_path} e escalador salvo em {scaler_path}")

    except Exception as e:
        print(f"Erro ao treinar os modelos: {str(e)}")



def start_scheduler():
    """
    Configura e inicia o agendador.
    """
    schedule.every().day.at(f"{START_TIME_HOUR:02d}:00").do(train_all_models)  # Treinar modelos

#    schedule.every().day.at(f"{START_TIME_HOUR:02d}:00").do(check_lottery_updates)
    print("Agendador iniciado. Aguardando tarefas...")
    train_all_models()

    while True:
        # Verifica se já passou das 10:00 mas ainda está antes das 11:00
        now = datetime.now(tz=SAO_PAULO_TZ)
        if now.hour >= START_TIME_HOUR and now < now.replace(hour=END_TIME_HOUR, minute=END_TIME_MINUTE, second=0, microsecond=0):
        #    train_all_models()
            check_lottery_updates()
        schedule.run_pending()
        time.sleep(1)  # Para evitar uso excessivo de CPU


def start_background_scheduler():
    """
    Inicia o agendador em uma thread separada.
    """
    
    thread = threading.Thread(target=start_scheduler, daemon=True)
    thread.start()
    print("Thread do agendador iniciada.\n")
