# Radiografía del S&P 500: perfiles financieros con ACP y K-Means

**Taller individual: ACP y K-Means** · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali
Autora: Sofia Haydar

🔗 **Aplicación:** https://videosp500.streamlit.app/
🎥 **Video de sustentación:** `<pega aquí el enlace del video>`

---

## 1. Contexto y pregunta

Los inversionistas suelen agrupar las acciones por **sector**, pero dos empresas del mismo sector pueden tener
perfiles financieros muy distintos (NVIDIA y una empresa madura de servicios de TI comparten sector GICS). A la vez,
el mercado habla de **estilos** (valor, crecimiento, dividendo, baja volatilidad) que cruzan los sectores.

El proyecto responde dos preguntas:

1. ¿Cuáles son las **dimensiones** que más diferencian a las empresas del S&P 500 según su valoración, rentabilidad,
   dividendo, tamaño y comportamiento reciente del precio?
2. ¿Existen **grupos** de empresas con perfiles parecidos, y en qué medida coinciden con los sectores?

**¿Por qué ACP y K-Means?** Las razones financieras están correlacionadas (las empresas grandes cotizan a múltiplos
altos; las que pagan dividendo son menos volátiles). El ACP resume esa redundancia y permite comprobar si aparecen
ejes conocidos en finanzas, como valor vs. crecimiento. No existe una etiqueta oficial de "estilo" para cada empresa,
así que el problema es no supervisado: K-Means busca grupos sin usar el sector, y después se compara el resultado con
el sector.

## 2. Datos

| | |
|---|---|
| **Fuente financiera** | Repositorio público [`datasets/s-and-p-500-companies-financials`](https://github.com/datasets/s-and-p-500-companies-financials) (datahub.io), que descarga a diario de **Yahoo Finance** con `yfinance`. |
| **Sector** | Clasificación GICS de la [lista de empresas del S&P 500 de Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies), incluida en el mismo repositorio. |
| **Fecha** | Snapshot del **17 de noviembre de 2026** (commit `482c023`). |
| **Observación** | Una empresa del S&P 500 (corte transversal). |
| **Cobertura** | 503 empresas en la lista → **419 en el análisis**. |

Los archivos crudos son `constituents-financials.csv` y `constituents_gics.csv`. Para volver a descargarlos:

```bash
python descargar_datos.py            # mismo snapshot del proyecto
python descargar_datos.py --ultimo   # datos más recientes (los resultados cambiarán)
```

### Variables construidas

| Variable | Definición |
|---|---|
| Tamaño | log10 de la capitalización bursátil |
| Rendimiento de utilidades (E/P) | UPA / precio (inverso del P/E; definido aun con pérdidas) |
| Rendimiento del dividendo | Dividendo anual / precio |
| Libro / precio (B/P) | 1 / (P/B), medida de "valor" de Fama-French |
| log(Precio / ventas) | Logaritmo del múltiplo P/S |
| Margen EBITDA | EBITDA / ventas, con ventas = capitalización / (P/S) |
| Amplitud del rango 52 semanas | (máximo − mínimo) / precio; aproxima la volatilidad |
| Posición en el rango 52 semanas | (precio − mínimo) / (máximo − mínimo); momentum |

### Limitaciones de la fuente

- Fotografía de un solo día: los múltiplos y las variables de 52 semanas dependen del precio de esa fecha.
- Yahoo Finance no está auditado y mezcla fechas de corte contable.
- La lista de Wikipedia del repositorio no está totalmente actualizada: 17 tickers ya no cotizan o tienen otro formato.
- EBITDA y ventas no son comparables para bancos y aseguradoras.

## 3. Metodología

1. **Limpieza** (`datos.py`): se eliminan 17 empresas sin cotización y se excluye el sector Financials (67), cuyos
   márgenes no son comparables y que en su mayoría no reporta EBITDA. No hay duplicados ni otros faltantes. `Symbol` y
   `Name` son identificadores y no entran al modelo.
2. **Transformaciones:** razones en lugar de niveles en dólares, inversos (E/P, B/P) para manejar pérdidas y patrimonio
   negativo, logaritmos para las variables asimétricas y **winsorización al 1 %** en cada cola.
3. **Estandarización** (media 0, desviación 1) antes del ACP y K-Means.
4. **ACP** (`modelo.py`): se conservan **4 componentes**, según el criterio de Kaiser (valores propios > 1) y el
   umbral del 75 % de varianza. Juntos explican el **77,8 %**.
5. **K-Means** sobre los 4 componentes, con K entre 2 y 10, evaluado con el método del codo, el coeficiente de silueta y
   la estabilidad entre semillas (ARI). Se elige **K = 5**.
6. **Interpretación:** los grupos se describen con las medianas de las variables originales y se cruzan con el sector GICS (V de Cramér).

Todos los parámetros (exclusión de Financials, winsorización, variables, número de componentes y K) se pueden cambiar
en la barra lateral de la app, que vuelve a ejecutar el procesamiento y los algoritmos.

## 4. Resultados principales

**Componentes** (con las 8 variables):

| Componente | Varianza | Interpretación |
|---|---|---|
| CP1 | 29,1 % | **Crecimiento/calidad vs. valor**: tamaño, P/S, margen y posición en el rango (+) frente a libro/precio (−) |
| CP2 | 19,9 % | **Volátil vs. defensivo**: amplitud del rango (+) frente a dividendo y posición en el rango (−) |
| CP3 | 15,8 % | **Rentabilidad castigada**: margen y dividendo altos con precio lejos del máximo |
| CP4 | 12,9 % | **Utilidades**: casi solo el rendimiento E/P |

**Grupos** (K = 5, silueta 0,272, ARI de estabilidad 0,74):

| Grupo | Empresas | Rasgos | Ejemplos |
|---|---|---|---|
| Calidad cerca de máximos | 163 | Múltiplos altos, baja volatilidad, cerca del máximo del año | AAPL, MSFT, AMZN, LLY, WMT |
| Valor de bajo margen | 116 | P/S y margen bajos, libro/precio alto | CVS, VLO, ELV, CMCSA, NOC |
| Defensivas de dividendo | 68 | Mayor dividendo y margen, menor volatilidad | NEE, VZ, T, MCD, COP |
| Crecimiento volátil | 64 | P/S muy alto, sin dividendo, rango amplio, lejos de máximos | NVDA, GOOGL, META, AVGO, TSLA |
| Con pérdidas contables | 8 | Utilidad negativa en los últimos 12 meses | F, KHC, PARA |

**Conclusiones:**
- El primer eje reproduce la distinción valor/crecimiento de Fama-French, pero hacen falta 4 componentes para llegar a
  ~78 % de la información: el perfil financiero no se reduce a un solo número.
- La estructura de grupos es **débil** (silueta ≈ 0,27): las empresas forman un continuo de estilos con zonas densas,
  no racimos separados. Hacer K-Means sobre las variables originales da una partición casi igual (ARI ≈ 0,91).
- Los grupos coinciden **solo en parte** con los sectores (V de Cramér = 0,45). Utilities y REITs se concentran en
  *Defensivas de dividendo*, pero tecnología se parte entre *Crecimiento volátil* y *Calidad*. **Diversificar por
  sector no garantiza diversificar por perfil.**
- **Limitación principal:** es un corte de un solo día; las variables de precio cambian con el ciclo del mercado.

## 5. Ejecutar el proyecto

```bash
git clone <url-del-repositorio>
cd sp500-perfiles
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Probado con Python 3.13.

**Despliegue:** en [share.streamlit.io](https://share.streamlit.io) → *Create app* → elegir el repositorio, la rama `main`
y el archivo `app.py`.

## 6. Estructura

```
├── app.py                         # Aplicación Streamlit (6 secciones)
├── datos.py                       # Carga, limpieza y construcción de variables
├── modelo.py                      # Estandarización, ACP, K-Means, silueta, estabilidad, V de Cramér
├── descargar_datos.py             # Descarga reproducible de los datos crudos
├── constituents-financials.csv    # Datos financieros (snapshot 17-nov-2026)
├── constituents_gics.csv          # Sector GICS de cada empresa
├── config.toml                    # Tema visual (opcional)
└── requirements.txt
```

## 7. Secciones de la aplicación

1. **Contexto:** pregunta, fuente, variables y limitaciones.
2. **Datos y preparación:** estructura, faltantes, bitácora de limpieza, descriptivos, histogramas y correlaciones.
3. **Componentes principales:** varianza, sedimentación, cargas y biplot interactivo.
4. **Segmentación K-Means:** codo, silueta, grupos, perfiles, cruce con sectores y silueta por empresa.
5. **Buscar una empresa:** grupo de cualquier empresa y sus 10 vecinos más cercanos (*peers*).
6. **Discusión y conclusiones.**

*Los grupos son una herramienta descriptiva, no una recomendación de inversión.*
