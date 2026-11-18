"""Carga de datos y construcción de variables (ratios) para el análisis.

Todo el procesamiento que usa la aplicación está aquí, para que el paso de los
datos crudos a la matriz que entra al ACP sea explícito y reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_FIN = RAIZ / "data" / "raw" / "constituents-financials.csv"
RUTA_GICS = RAIZ / "data" / "raw" / "constituents_gics.csv"

# Variables analíticas: nombre interno -> (etiqueta legible, descripción)
VARIABLES = {
    "tamano": ("Tamaño (log10 cap. bursátil)",
               "Logaritmo base 10 de la capitalización bursátil en USD. 11 ≈ 100 mil millones."),
    "rend_utilidades": ("Rendimiento de utilidades (E/P)",
                        "Utilidad por acción / precio. Es el inverso del P/E y está definido aun con pérdidas."),
    "rend_dividendo": ("Rendimiento del dividendo",
                       "Dividendos anuales / precio."),
    "libro_precio": ("Libro / precio (B/P)",
                     "Inverso de P/B. Medida clásica de 'valor' (Fama-French); admite patrimonio negativo."),
    "log_precio_ventas": ("log(Precio / ventas)",
                          "Logaritmo natural del múltiplo P/S. Alto = el mercado paga mucho por cada dólar vendido."),
    "margen_ebitda": ("Margen EBITDA",
                      "EBITDA / ventas, con ventas = cap. bursátil / (P/S). Mide rentabilidad operativa."),
    "rango_52s": ("Amplitud del rango 52 semanas",
                  "(Máximo − mínimo de 52 semanas) / precio. Aproxima la volatilidad del último año."),
    "posicion_52s": ("Posición en el rango 52 semanas",
                     "(Precio − mínimo) / (máximo − mínimo). 1 = en máximos del año; 0 = en mínimos (momentum)."),
}


def cargar_crudos() -> pd.DataFrame:
    """Une los datos financieros (Yahoo Finance vía datahub) con el sector GICS (Wikipedia)."""
    fin = pd.read_csv(RUTA_FIN)
    gics = pd.read_csv(RUTA_GICS)[["Symbol", "GICS Sector", "GICS Sub-Industry"]]
    df = fin.drop(columns=["SEC Filings", "Sector"]).merge(gics, on="Symbol", how="left")
    return df.rename(columns={"GICS Sector": "Sector", "GICS Sub-Industry": "Subindustria"})


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula los ratios a partir de las columnas originales."""
    out = df[["Symbol", "Name", "Sector", "Subindustria"]].copy()
    ventas = df["Market Cap"] / df["Price/Sales"]
    out["tamano"] = np.log10(df["Market Cap"])
    out["rend_utilidades"] = df["Earnings/Share"] / df["Price"]
    out["rend_dividendo"] = df["Dividend Yield"]
    out["libro_precio"] = 1 / df["Price/Book"]
    out["log_precio_ventas"] = np.log(df["Price/Sales"])
    out["margen_ebitda"] = df["EBITDA"] / ventas
    rango = df["52 Week High"] - df["52 Week Low"]
    out["rango_52s"] = rango / df["Price"]
    out["posicion_52s"] = (df["Price"] - df["52 Week Low"]) / rango
    out["Market Cap"] = df["Market Cap"]
    return out


def preparar(excluir_financieras: bool = True, winsor: float = 0.01):
    """Aplica las decisiones de calidad y devuelve (base_limpia, bitacora).

    - Excluye el sector Financials (opcional): EBITDA, ventas y márgenes no son
      comparables para bancos y aseguradoras, y la mayoría no reporta EBITDA.
    - Elimina empresas sin cotización (tickers que salieron del índice o con otro formato).
    - Elimina empresas con alguna variable faltante (sobre todo EBITDA).
    - Winsoriza cada variable en los percentiles `winsor` y 1 - `winsor`.
    """
    crudos = cargar_crudos()
    bitacora = [("Empresas en la lista del S&P 500", len(crudos))]

    dup = crudos["Symbol"].duplicated().sum()
    crudos = crudos.drop_duplicates("Symbol")
    bitacora.append(("Tickers duplicados eliminados", int(dup)))

    sin_precio = crudos["Price"].isna() | crudos["Market Cap"].isna()
    bitacora.append(("Sin cotización en Yahoo (fuera del índice o ticker con otro formato)", int(sin_precio.sum())))
    crudos = crudos[~sin_precio]

    if excluir_financieras:
        fin = crudos["Sector"].eq("Financials")
        bitacora.append(("Sector Financials excluido", int(fin.sum())))
        crudos = crudos[~fin]

    base = construir_variables(crudos)
    cols = list(VARIABLES)
    base = base.replace([np.inf, -np.inf], np.nan)
    falt = base[cols].isna().any(axis=1)
    bitacora.append(("Con al menos una variable faltante (principalmente EBITDA)", int(falt.sum())))
    base = base[~falt].reset_index(drop=True)

    if winsor > 0:
        for c in cols:
            lo, hi = base[c].quantile([winsor, 1 - winsor])
            base[c] = base[c].clip(lo, hi)
    bitacora.append(("Empresas que entran al análisis", len(base)))
    return base, pd.DataFrame(bitacora, columns=["Paso", "Empresas"])
