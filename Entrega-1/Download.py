import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://transparenciachc.blob.core.windows.net/lic-da/{year}-{month}.zip"
OUTPUT_DIR = "descargas"

START_YEAR = 2007
END_YEAR = 2025

MAX_WORKERS = 32      
TIMEOUT = 20
CHUNK_SIZE = 8192


def create_session():
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def descargar_archivo(session, year, month):
    url = BASE_URL.format(year=year, month=month)
    filename = f"{year}-{month}.zip"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        return f"[EXISTE] {filename}"

    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as response:
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                return f"[OK] {filename}"
            elif response.status_code == 404:
                return f"[NO EXISTE] {filename}"
            else:
                return f"[HTTP {response.status_code}] {filename}"

    except requests.exceptions.RequestException as e:
        return f"[ERROR] {filename} -> {e}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session = create_session()

    tareas = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                tareas.append(executor.submit(descargar_archivo, session, year, month))

        for future in as_completed(tareas):
            print(future.result())

    print("Descarga paralela finalizada.")


if __name__ == "__main__":
    main()