import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
from webdriver_manager.chrome import ChromeDriverManager
from django.db.models import F

# Configurar driver do Selenium
def configurar_driver(download_dir):
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "safebrowsing.enabled": True,
    })
    chrome_options.add_argument("--enable-gpu")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--disable-accelerated-2d-canvas")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


# Função para baixar arquivo
def baixar_arquivo(driver, url, xpath, download_dir):
    try:
        driver.get(url)
        driver.maximize_window()
        print(f"Acessando URL: {url}")

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        elemento = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", elemento)
        time.sleep(1)

        # Tentar obter o atributo href
        href = elemento.get_attribute("href")
        if href:
            nome_arquivo = href.split("/")[-1]
            caminho_arquivo = os.path.join(download_dir, nome_arquivo)

            # Verifica se o arquivo já existe e apaga antes de baixar
            if os.path.exists(caminho_arquivo):
                os.remove(caminho_arquivo)

            driver.execute_script("arguments[0].click();", elemento)
            print(f"Download iniciado para: {href}")
        else:
            print("Elemento não possui atributo 'href'. Tentando clique direto.")
            driver.execute_script("arguments[0].click();", elemento)

        time.sleep(5)
    except Exception as e:
        print(f"Erro ao baixar arquivo de {url}: {e}")



# Função para converter arquivos XLSX para CSV
def converter_para_csv(pasta_download):
    try:
        for arquivo in os.listdir(pasta_download):
            if arquivo.endswith(".xlsx"):
                caminho_arquivo = os.path.join(pasta_download, arquivo)
                novo_caminho = os.path.splitext(caminho_arquivo)[0] + ".csv"

                # Converte usando pandas
                df = pd.read_excel(caminho_arquivo)
                df.to_csv(novo_caminho, index=False)
                print(f"Convertido: {caminho_arquivo} -> {novo_caminho}")

                # Remove o arquivo XLSX
                os.remove(caminho_arquivo)
    except Exception as e:
        print(f"Erro ao converter arquivos: {e}")


# Função principal
def executar_script():
    pasta_download = os.path.join(os.getcwd(), "loterias")
    Path(pasta_download).mkdir(exist_ok=True)  # Cria a pasta se não existir

    driver = configurar_driver(pasta_download)

    try:
        # URLs e seus respectivos XPaths
        sites = [
            ("https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx", '//*[@id="btnResultados"]'),
            ("https://loterias.caixa.gov.br/Paginas/Lotofacil.aspx", '//*[@id="btnResultados"]'),
            ("https://loterias.caixa.gov.br/Paginas/Quina.aspx", '//*[@id="btnResultados"]'),
        ]

        for url, xpath in sites:
            baixar_arquivo(driver, url, xpath, pasta_download)

        converter_para_csv(pasta_download)
        processar_dados(pasta_download)

    finally:
        driver.quit()


def processar_dados(pasta_download):
    from lottery_app.models import LotteryGame  # Importa dentro da função para evitar erro
    import pandas as pd
    from django.core.files import File

    for arquivo in os.listdir(pasta_download):
        if arquivo.endswith(".csv"):
            nome_loteria = arquivo.split(".")[0]
            caminho_arquivo = os.path.join(pasta_download, arquivo)
            try:
                # Lê o arquivo CSV
                df = pd.read_csv(caminho_arquivo)

                # Obtém o último concurso
                ultimo_concurso = df["Concurso"].max()

                # Combina as colunas que contêm "Bola" em uma string
                colunas_bolas = [col for col in df.columns if "Bola" in col]
                ultima_linha = df[df["Concurso"] == ultimo_concurso]
                numeros_sorteados = ultima_linha[colunas_bolas].values.flatten()
                sorteados = ",".join(f"{int(num):02}" for num in numeros_sorteados)

                # Atualiza o registro no banco de dados
                jogo = LotteryGame.objects.filter(name__iexact=nome_loteria).first()
                if jogo:
                    # Apaga o arquivo antigo, se existir
                    if jogo.historical_data and jogo.historical_data.name:
                        jogo.historical_data.delete(save=False)

                    # Substitui o arquivo antigo pelo novo
                    with open(caminho_arquivo, 'rb') as novo_arquivo:
                        jogo.historical_data.save(arquivo, File(novo_arquivo), save=False)

                    # Atualiza as outras informações
                    jogo.concurso = ultimo_concurso
                    jogo.sorteados = sorteados
                    jogo.save()
                    print(f"Dados do jogo '{nome_loteria}' atualizados com sucesso.")
                else:
                    print(f"Jogo '{nome_loteria}' não encontrado no banco de dados.")
            except Exception as e:
                print(f"Erro ao processar arquivo {arquivo}: {e}")


if __name__ == "__main__":
    executar_script()
