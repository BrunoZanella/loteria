import schedule
import time
from datetime import datetime

def tarefa_principal():
    """
    Função principal da tarefa agendada.
    """
    print(f"Tarefa executada às {datetime.now()}")

def start_scheduler():
    """
    Inicia o agendador e mantém a execução contínua.
    """
    schedule.every().day.at("19:00").do(tarefa_principal)

    print("Agendador do tasks iniciado. Aguardando o horário...")
    while True:
        schedule.run_pending()
        time.sleep(1)  # Reduz uso de CPU
