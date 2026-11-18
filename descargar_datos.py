"""Descarga los datos crudos usados en el proyecto.

Por defecto descarga el snapshot exacto usado en el análisis (commit 482c023 del
repositorio datasets/s-and-p-500-companies-financials, 17-nov-2026). Con --ultimo
descarga la versión más reciente (el repositorio se actualiza a diario desde Yahoo Finance),
lo que puede cambiar los resultados.

Uso:
    python scripts/descargar_datos.py            # snapshot del proyecto
    python scripts/descargar_datos.py --ultimo   # datos más recientes
"""
import argparse
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

COMMIT = "482c0232f216525ea78689a573a7352c77fd8f6d"
REPO = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials"
DESTINO = Path(__file__).resolve().parents[1] / "data" / "raw"


def leer(url: str) -> str:
    with urlopen(url) as r:
        return r.read().decode("utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ultimo", action="store_true", help="Usar la versión más reciente en lugar del snapshot")
    ref = "main" if p.parse_args().ultimo else COMMIT
    DESTINO.mkdir(parents=True, exist_ok=True)

    # 1) Datos financieros (Yahoo Finance vía datahub)
    fin = leer(f"{REPO}/{ref}/data/constituents-financials.csv")
    (DESTINO / "constituents-financials.csv").write_text(fin, encoding="utf-8")

    # 2) Sector GICS: tabla 'constituents' de la página de Wikipedia guardada en el repositorio
    html = leer(f"{REPO}/{ref}/scripts/List_of_S%2526P_500_companies.html")
    tabla = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    tabla[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].to_csv(
        DESTINO / "constituents_gics.csv", index=False)
    print(f"Datos guardados en {DESTINO} (referencia: {ref[:7]})")


if __name__ == "__main__":
    main()
