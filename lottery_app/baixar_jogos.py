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


# Configurar driver do Selenium
def configurar_driver(download_dir):
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "safebrowsing.enabled": True,
    })
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-notifications")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


# Função para baixar arquivo
def baixar_arquivo(driver, url, xpath, download_dir):
    try:
        driver.get(url)
        print(f"Acessando URL: {url}")

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        elemento = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", elemento)
        time.sleep(1)

        nome_arquivo = elemento.get_attribute("href").split("/")[-1]
        caminho_arquivo = os.path.join(download_dir, nome_arquivo)

        # Verifica se o arquivo já existe e apaga antes de baixar
        if os.path.exists(caminho_arquivo):
            os.remove(caminho_arquivo)

        driver.execute_script("arguments[0].click();", elemento)
        print(f"Download iniciado para: {url}")
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
            ("https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx", '//*[@id="resultados"]/div/ul/li/a'),
            ("https://loterias.caixa.gov.br/Paginas/Lotofacil.aspx", '//*[@id="resultados"]/div/ul/li/a'),
            ("https://loterias.caixa.gov.br/Paginas/Quina.aspx", '//*[@id="resultados"]/div/ul/li/a'),
        ]

        for url, xpath in sites:
            baixar_arquivo(driver, url, xpath, pasta_download)

        converter_para_csv(pasta_download)
    finally:
        driver.quit()


if __name__ == "__main__":
    executar_script()
