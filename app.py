"""Radiografía del S&P 500: perfiles financieros con ACP y K-Means.

Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.datos import VARIABLES, cargar_crudos, preparar
from src.modelo import (ajustar_acp, ajustar_kmeans, estabilidad, estandarizar, evaluar_k,
                        nombrar_grupos, sugerir_componentes, v_de_cramer)

st.set_page_config(page_title="Radiografía del S&P 500", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- estilo y paleta
# Paleta categórica validada para daltonismo (orden fijo). Con más de 3 grupos en un
# diagrama de dispersión se agrega un símbolo distinto por grupo como segunda codificación.
CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SIMBOLOS = ["circle", "square", "diamond", "triangle-up", "x", "cross", "star", "hexagon"]
DIVERGENTE = [[0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec8b8a"], [1, "#b02a2a"]]
SECUENCIAL = ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
ETIQ = {k: v[0] for k, v in VARIABLES.items()}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.nota {background: #f4f6fa; border-left: 4px solid #2a78d6; padding: .8rem 1rem;
       border-radius: 4px; margin: .6rem 0 1rem 0; font-size: .95rem;}
.kpi {font-size: 2rem; font-weight: 700; line-height: 1.1;}
.kpi-sub {color: #52514e; font-size: .85rem;}
</style>""", unsafe_allow_html=True)


def nota(texto: str):
    st.markdown(f"<div class='nota'>{texto}</div>", unsafe_allow_html=True)


def estilo(fig, alto=430):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor="white",
                      font=dict(size=13), title_font=dict(size=16), legend_title_text="")
    fig.update_xaxes(showgrid=True, gridcolor="#ececec", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ececec", zeroline=False)
    return fig


# ---------------------------------------------------------------- cálculos en caché
@st.cache_data
def datos(excluir_fin: bool, winsor: float):
    return preparar(excluir_financieras=excluir_fin, winsor=winsor)


@st.cache_data
def acp_cache(excluir_fin, winsor, cols):
    base, _ = datos(excluir_fin, winsor)
    X, _ = estandarizar(base, list(cols))
    _, var, cargas, punt = ajustar_acp(X, list(cols))
    return X, var, cargas, punt


@st.cache_data
def k_cache(excluir_fin, winsor, cols, n_comp):
    _, _, _, punt = acp_cache(excluir_fin, winsor, cols)
    return evaluar_k(punt.iloc[:, :n_comp].to_numpy())


@st.cache_data
def segmentar(excluir_fin, winsor, cols, n_comp, k):
    base, _ = datos(excluir_fin, winsor)
    X, _, _, punt = acp_cache(excluir_fin, winsor, cols)
    Z = punt.iloc[:, :n_comp].to_numpy()
    km, sil = ajustar_kmeans(Z, k)
    z = pd.DataFrame(X, columns=list(cols))
    perfil_z = z.groupby(km.labels_).median()
    nombres = nombrar_grupos(perfil_z)
    grupos = pd.Series(km.labels_).map(nombres)
    estab = estabilidad(Z, k, km.labels_)
    return grupos.to_numpy(), sil, perfil_z.rename(index=nombres), estab


# ---------------------------------------------------------------- barra lateral
st.sidebar.title("📊 Radiografía del S&P 500")
pagina = st.sidebar.radio("Secciones", [
    "1. Contexto", "2. Datos y preparación", "3. Componentes principales",
    "4. Segmentación K-Means", "5. Buscar una empresa", "6. Discusión y conclusiones"])

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros del análisis")
st.sidebar.caption("Cambiarlos vuelve a ejecutar la limpieza, el ACP y K-Means.")
excluir_fin = st.sidebar.checkbox("Excluir el sector Financials", value=True,
                                  help="Bancos y aseguradoras no reportan EBITDA ni ventas comparables.")
winsor = st.sidebar.slider("Winsorización (% en cada cola)", 0.0, 5.0, 1.0, 0.5,
                           help="Recorta valores extremos al percentil indicado para que no dominen la varianza.") / 100
cols = tuple(st.sidebar.multiselect("Variables del modelo", list(VARIABLES), default=list(VARIABLES),
                                    format_func=lambda c: ETIQ[c]))
if len(cols) < 4:
    st.sidebar.error("Selecciona al menos 4 variables.")
    st.stop()

base, bitacora = datos(excluir_fin, winsor)
X, varianza, cargas, puntajes = acp_cache(excluir_fin, winsor, cols)
sug = sugerir_componentes(varianza)
n_comp = st.sidebar.slider("Componentes para K-Means", 2, len(cols), min(max(sug["kaiser"], 2), len(cols)),
                           help=f"Kaiser sugiere {sug['kaiser']}; para ≥75 % de varianza se necesitan {sug['acumulado']}.")
k = st.sidebar.slider("Número de grupos (K)", 2, 10, 5)

grupos, sil, perfil_z, estab = segmentar(excluir_fin, winsor, cols, n_comp, k)
base = base.assign(Grupo=grupos, Silueta=sil)
orden_grupos = sorted(base["Grupo"].unique())
color_grupo = {g: CATEGORICA[i % 8] for i, g in enumerate(orden_grupos)}
simbolo_grupo = {g: SIMBOLOS[i % 8] for i, g in enumerate(orden_grupos)}

st.sidebar.markdown("---")
st.sidebar.caption("Datos: Yahoo Finance vía datahub.io (snapshot del 17-nov-2026) · Sector GICS: Wikipedia.")

# ================================================================= 1. CONTEXTO
if pagina == "1. Contexto":
    st.title("¿Qué tipos de empresa hay realmente en el S&P 500?")
    st.markdown("#### Perfiles financieros con Análisis de Componentes Principales y K-Means")

    c1, c2, c3, c4 = st.columns(4)
    for c, v, s in [(c1, len(base), "empresas analizadas"), (c2, len(cols), "variables cuantitativas"),
                    (c3, f"{varianza['Acumulada'].iloc[n_comp - 1]:.0%}", f"varianza en {n_comp} componentes"),
                    (c4, k, "grupos identificados")]:
        c.markdown(f"<div class='kpi'>{v}</div><div class='kpi-sub'>{s}</div>", unsafe_allow_html=True)

    st.markdown("### La pregunta")
    st.markdown("""
Los inversionistas suelen clasificar las acciones por **sector** (tecnología, salud, energía...).
Pero dos empresas del mismo sector pueden ser financieramente muy distintas: NVIDIA y una empresa
de servicios de TI madura comparten sector GICS y casi nada más. Al mismo tiempo, el mercado
habla de **estilos** (valor, crecimiento, dividendo, baja volatilidad) que cruzan los sectores.

Este proyecto pregunta:
1. **¿Cuáles son las dimensiones que más diferencian a las empresas del S&P 500** cuando se miran
   su valoración, rentabilidad, dividendo, tamaño y comportamiento reciente del precio?
2. **¿Existen grupos de empresas con perfiles financieros parecidos**, y en qué medida coinciden
   con los sectores tradicionales?

Responderlo sirve para construir carteras diversificadas por *perfil* y no solo por sector,
para encontrar empresas comparables (*peers*) en una valoración, y para entender qué estilo de
inversión está detrás de una acción.
""")

    st.markdown("### ¿Por qué ACP y K-Means?")
    st.markdown("""
- Las ocho razones financieras están **correlacionadas entre sí** (las empresas grandes tienden a
  cotizar a múltiplos altos; las que pagan dividendo tienden a ser menos volátiles). El **ACP**
  resume esa información redundante en pocas dimensiones y permite ver si emergen ejes conocidos
  en finanzas, como *valor vs. crecimiento*.
- No existe una etiqueta de "estilo" oficial para cada empresa, así que el problema es **no
  supervisado**. **K-Means** busca grupos de empresas cercanas entre sí en el espacio de
  componentes, sin usar el sector. Luego se compara el resultado con el sector para responder la
  segunda pregunta.
- Aplicar K-Means sobre los componentes, y no sobre las variables originales, evita que las
  dimensiones redundantes pesen doble en las distancias y elimina ruido de los componentes menores.
""")

    st.markdown("### Fuente y cobertura")
    st.markdown("""
- **Datos financieros:** repositorio público [`datasets/s-and-p-500-companies-financials`](https://github.com/datasets/s-and-p-500-companies-financials)
  de datahub.io, que descarga a diario de **Yahoo Finance** (vía `yfinance`) precio, utilidad por acción, P/E,
  dividendo, capitalización, EBITDA, P/S, P/B y rango de 52 semanas de cada empresa del índice.
  Se usa el snapshot del **17 de noviembre de 2026** (commit `482c023`).
- **Sector:** clasificación GICS de la [lista de empresas del S&P 500 en Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies),
  incluida en el mismo repositorio.
- **Unidad de observación:** una empresa del S&P 500 (corte transversal, una fecha).
""")

    st.markdown("### Variables")
    st.dataframe(pd.DataFrame([(v[0], v[1]) for v in VARIABLES.values()], columns=["Variable", "Definición"]),
                 hide_index=True, width="stretch")

    st.markdown("### Limitaciones de la fuente")
    st.markdown("""
- Es una **fotografía de un solo día**: las variables de precio (múltiplos, posición en el rango de 52 semanas)
  cambian con el mercado, así que los grupos de otra fecha pueden ser distintos.
- Yahoo Finance usa cifras de los **últimos doce meses** y no audita los datos; hay valores atípicos
  (p. ej. P/E de miles) que se tratan con winsorización.
- La lista de empresas de Wikipedia en el repositorio no está totalmente actualizada: algunos tickers ya
  salieron del índice y no tienen cotización.
- Las razones basadas en EBITDA y ventas **no aplican a bancos y aseguradoras**, por eso se excluye ese sector
  por defecto.
""")

# ================================================================= 2. DATOS
elif pagina == "2. Datos y preparación":
    st.title("Datos y preparación")
    crudos = cargar_crudos()

    st.subheader("Estructura de la base original")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", len(crudos))
    c2.metric("Columnas", crudos.shape[1])
    c3.metric("Tickers duplicados", int(crudos["Symbol"].duplicated().sum()))
    with st.expander("Ver datos originales"):
        st.dataframe(crudos, width="stretch")
    tipos = pd.DataFrame({"Tipo": crudos.dtypes.astype(str), "Faltantes": crudos.isna().sum(),
                          "% faltante": (crudos.isna().mean() * 100).round(1)})
    st.dataframe(tipos, width="stretch")
    nota("Las 17 empresas sin precio corresponden a tickers de la lista de Wikipedia que ya no cotizan "
         "(adquisiciones como Ansys, Hess o Juniper) o que Yahoo escribe distinto (BRK.B, BF.B). "
         "El EBITDA falta sobre todo en el sector Financials. <b>Symbol</b> y <b>Name</b> son "
         "identificadores y no entran al modelo.")

    st.subheader("Decisiones de limpieza")
    st.dataframe(bitacora, hide_index=True, width="stretch")
    st.markdown(f"""
- **Ratios en lugar de niveles.** Precio, EBITDA o capitalización en dólares medirían sobre todo el tamaño.
  Se construyen razones comparables entre empresas (ver *Contexto*).
- **E/P en vez de P/E y B/P en vez de P/B.** El P/E no existe cuando hay pérdidas y explota cerca de cero;
  su inverso (rendimiento de utilidades) es continuo y admite valores negativos. Igual con el patrimonio
  negativo de empresas que recompran muchas acciones.
- **Logaritmos** en tamaño y P/S, que tienen distribuciones muy asimétricas.
- **Winsorización al {winsor:.1%}** en cada cola (ajustable en la barra lateral) para que unos pocos
  valores extremos no dominen la varianza del ACP ni los centroides.
- **Estandarización (media 0, desviación 1)** antes del ACP y K-Means: ambos dependen de distancias y
  varianzas, y las variables están en escalas distintas (el tamaño va de 9,7 a 12,5; el dividendo de 0 a 0,06).
""")

    st.subheader("Estadísticas descriptivas (variables construidas)")
    desc = base[list(cols)].describe().T.rename(index=ETIQ)
    st.dataframe(desc.style.format("{:.3f}"), width="stretch")

    st.subheader("Distribución de cada variable")
    var_hist = st.selectbox("Variable", cols, format_func=lambda c: ETIQ[c])
    fig = px.histogram(base, x=var_hist, nbins=40, color_discrete_sequence=[CATEGORICA[0]],
                       title=f"Distribución de {ETIQ[var_hist].lower()} ({len(base)} empresas)")
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(xaxis_title=ETIQ[var_hist], yaxis_title="Número de empresas", bargap=0.05)
    st.plotly_chart(estilo(fig, 380), width="stretch")
    st.caption(VARIABLES[var_hist][1])

    st.subheader("Matriz de correlaciones")
    corr = base[list(cols)].corr().rename(index=ETIQ, columns=ETIQ)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=DIVERGENTE, zmin=-1, zmax=1,
                    title="Correlación de Pearson entre las variables (después de winsorizar)")
    st.plotly_chart(estilo(fig, 560), width="stretch")
    nota("Las correlaciones más fuertes justifican el ACP: <b>tamaño</b> y <b>P/S</b> se mueven juntos y en sentido "
         "contrario al <b>libro/precio</b> (las empresas grandes cotizan caro frente a su patrimonio), y la "
         "<b>amplitud del rango de 52 semanas</b> se opone a la <b>posición en el rango</b>: las acciones más "
         "volátiles del año son, en esta fecha, las que más cayeron desde sus máximos. Ninguna correlación es "
         "cercana a 1, así que ninguna variable es una copia de otra.")

# ================================================================= 3. ACP
elif pagina == "3. Componentes principales":
    st.title("Análisis de Componentes Principales")
    st.markdown(f"ACP sobre las **{len(cols)} variables estandarizadas** de **{len(base)} empresas**.")

    st.subheader("Varianza explicada")
    tabla = varianza.copy()
    st.dataframe(tabla.style.format({"Valor propio": "{:.2f}", "Varianza explicada": "{:.1%}", "Acumulada": "{:.1%}"}),
                 hide_index=True, width="stretch")

    fig = go.Figure()
    fig.add_bar(x=varianza["Componente"], y=varianza["Valor propio"], name="Valor propio",
                marker_color=CATEGORICA[0], hovertemplate="%{x}: %{y:.2f}<extra></extra>")
    fig.add_hline(y=1, line_dash="dash", line_color="#52514e",
                  annotation_text="Criterio de Kaiser (valor propio = 1)", annotation_position="top right")
    fig.update_layout(title="Gráfico de sedimentación", xaxis_title="Componente", yaxis_title="Valor propio",
                      showlegend=False)
    c1, c2 = st.columns(2)
    c1.plotly_chart(estilo(fig, 380), width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=varianza["Componente"], y=varianza["Acumulada"], mode="lines+markers",
                    line=dict(color=CATEGORICA[0], width=2), marker=dict(size=9),
                    hovertemplate="%{x}: %{y:.1%}<extra></extra>")
    fig.add_hline(y=0.75, line_dash="dash", line_color="#52514e", annotation_text="75 %",
                  annotation_position="bottom right")
    fig.update_layout(title="Varianza explicada acumulada", xaxis_title="Componente",
                      yaxis_title="Proporción acumulada", yaxis_tickformat=".0%", showlegend=False)
    c2.plotly_chart(estilo(fig, 380), width="stretch")

    nota(f"<b>¿Cuántos componentes conservar?</b> El criterio de Kaiser sugiere <b>{sug['kaiser']}</b> "
         f"(valores propios mayores que 1: una componente que resume menos que una variable original no aporta "
         f"como resumen). Para superar el 75 % de la varianza se necesitan <b>{sug['acumulado']}</b>. "
         f"Con {n_comp} componentes (elección actual) se conserva el "
         f"<b>{varianza['Acumulada'].iloc[n_comp - 1]:.1%}</b> de la información. El codo del gráfico no es "
         f"nítido: la varianza decae de forma gradual, señal de que los datos no se reducen a uno o dos ejes.")

    st.subheader("Cargas: ¿qué significa cada componente?")
    n_mostrar = min(max(n_comp, 2), len(cols))
    car = cargas.iloc[:, :n_mostrar].rename(index=ETIQ)
    fig = px.imshow(car, text_auto=".2f", color_continuous_scale=DIVERGENTE, zmin=-1, zmax=1, aspect="auto",
                    title=f"Cargas de las variables en los primeros {n_mostrar} componentes")
    fig.update_layout(xaxis_title="Componente", yaxis_title="")
    st.plotly_chart(estilo(fig, 470), width="stretch")
    if set(cols) == set(VARIABLES):
        st.markdown("""
**Lectura de las cargas (con las 8 variables):**
- **CP1 – Crecimiento/calidad vs. valor.** Cargas positivas en P/S, tamaño, margen EBITDA y posición en el rango;
  negativa en libro/precio. Un puntaje alto describe empresas grandes, muy rentables y caras frente a sus
  ventas y su patrimonio (el estilo *growth*); un puntaje bajo, empresas baratas frente a sus activos (*value*).
- **CP2 – Riesgo reciente vs. perfil defensivo.** Positiva en la amplitud del rango de 52 semanas; negativa
  en dividendo y posición en el rango. Separa acciones volátiles que han caído desde sus máximos de
  acciones estables que pagan dividendo.
- **CP3 – Rentabilidad castigada.** Margen EBITDA y dividendo altos con posición baja en el rango:
  negocios rentables cuyo precio se ha rezagado en el año.
- **CP4 – Utilidades.** Casi exclusivamente el rendimiento de utilidades (carga ≈ 0,97). Recoge la información
  de las ganancias contables, que es casi independiente del resto; su valor propio apenas supera 1.
""")
    else:
        st.info("Cambiaste las variables: interpreta las cargas con el mapa de calor (|carga| > 0,4 indica una variable relevante).")

    st.subheader("Empresas en el espacio de componentes")
    c1, c2, c3 = st.columns(3)
    nombres_cp = list(puntajes.columns)
    eje_x = c1.selectbox("Eje horizontal", nombres_cp, index=0)
    eje_y = c2.selectbox("Eje vertical", nombres_cp, index=1)
    color_por = c3.selectbox("Colorear por", ["Ninguno"] + list(cols), format_func=lambda c: ETIQ.get(c, c))
    datos_plot = pd.concat([base[["Symbol", "Name", "Sector"] + list(cols)], puntajes], axis=1)
    pct = varianza.set_index("Componente")["Varianza explicada"]
    fig = px.scatter(datos_plot, x=eje_x, y=eje_y, hover_name="Name",
                     hover_data={"Symbol": True, "Sector": True, eje_x: ":.2f", eje_y: ":.2f"},
                     color=None if color_por == "Ninguno" else color_por,
                     color_continuous_scale=SECUENCIAL,
                     labels={c: ETIQ[c] for c in cols},
                     title=f"{eje_x} vs {eje_y}: resumen del {pct[eje_x] + pct[eje_y]:.1%} de la varianza total")
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white"),
                                  **({} if color_por != "Ninguno" else {"color": CATEGORICA[0]})))
    # Biplot: flechas con las cargas
    if st.checkbox("Mostrar vectores de cargas (biplot)", value=True):
        esc = float(np.abs(datos_plot[[eje_x, eje_y]]).max().max()) * 0.9
        for var in cols:
            lx, ly = cargas.loc[var, eje_x] * esc, cargas.loc[var, eje_y] * esc
            fig.add_annotation(x=lx, y=ly, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#52514e")
            fig.add_annotation(x=lx * 1.08, y=ly * 1.08, text=ETIQ[var].split(" (")[0], showarrow=False,
                               font=dict(size=11, color="#0b0b0b"), bgcolor="rgba(255,255,255,.7)")
    fig.update_layout(xaxis_title=f"{eje_x} ({pct[eje_x]:.1%})", yaxis_title=f"{eje_y} ({pct[eje_y]:.1%})")
    st.plotly_chart(estilo(fig, 620), width="stretch")
    nota(f"Este plano solo resume el <b>{pct[eje_x] + pct[eje_y]:.1%}</b> de la variación: empresas que se ven "
         "cercanas aquí pueden diferir en los componentes que no se muestran. Pasa el cursor para ver cada empresa.")

# ================================================================= 4. K-MEANS
elif pagina == "4. Segmentación K-Means":
    st.title("Segmentación con K-Means")
    st.markdown(f"K-Means sobre los **{n_comp} primeros componentes** "
                f"({varianza['Acumulada'].iloc[n_comp - 1]:.1%} de la varianza), con 50 inicializaciones.")

    st.subheader("¿Cuántos grupos?")
    ev = k_cache(excluir_fin, winsor, cols, n_comp)
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=ev["K"], y=ev["Inercia"], mode="lines+markers", line=dict(color=CATEGORICA[0], width=2),
                    marker=dict(size=9), hovertemplate="K=%{x}: %{y:.0f}<extra></extra>")
    fig.add_vline(x=k, line_dash="dot", line_color="#52514e")
    fig.update_layout(title="Método del codo", xaxis_title="Número de grupos (K)",
                      yaxis_title="Inercia (suma de distancias² al centroide)", showlegend=False)
    c1.plotly_chart(estilo(fig, 380), width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=ev["K"], y=ev["Silueta"], mode="lines+markers", line=dict(color=CATEGORICA[1], width=2),
                    marker=dict(size=9), hovertemplate="K=%{x}: %{y:.3f}<extra></extra>")
    fig.add_vline(x=k, line_dash="dot", line_color="#52514e")
    fig.update_layout(title="Coeficiente de silueta promedio", xaxis_title="Número de grupos (K)",
                      yaxis_title="Silueta (−1 a 1)", showlegend=False)
    c2.plotly_chart(estilo(fig, 380), width="stretch")
    mejor = ev.loc[ev["Silueta"].idxmax()]
    nota(f"La inercia cae sin un codo marcado. La silueta más alta en este rango se obtiene con "
         f"<b>K = {int(mejor['K'])}</b> ({mejor['Silueta']:.3f}). Con K = {k} (elección actual) la silueta es "
         f"<b>{sil.mean():.3f}</b>. Valores entre 0,25 y 0,30 indican una <b>estructura débil</b>: hay grupos, pero "
         f"sus fronteras se tocan. La estabilidad (ARI entre soluciones con semillas distintas) es "
         f"<b>{estab:.2f}</b>; cerca de 1 significa que la partición no depende del punto de partida.")

    st.subheader("Los grupos en el plano de los dos primeros componentes")
    datos_plot = pd.concat([base[["Symbol", "Name", "Sector", "Grupo"]], puntajes.iloc[:, :2]], axis=1)
    pct = varianza["Varianza explicada"]
    fig = px.scatter(datos_plot, x="CP1", y="CP2", color="Grupo", symbol="Grupo", hover_name="Name",
                     hover_data={"Symbol": True, "Sector": True, "Grupo": False},
                     color_discrete_map=color_grupo, symbol_map=simbolo_grupo,
                     category_orders={"Grupo": orden_grupos},
                     title=f"Grupos de K-Means proyectados en CP1 y CP2 ({pct.iloc[0] + pct.iloc[1]:.1%} de la varianza)")
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
    fig.update_layout(xaxis_title=f"CP1 ({pct.iloc[0]:.1%})", yaxis_title=f"CP2 ({pct.iloc[1]:.1%})")
    if set(cols) == set(VARIABLES):
        fig.update_layout(xaxis_title=f"CP1: valor (−) ↔ crecimiento/calidad (+)  ({pct.iloc[0]:.1%})",
                          yaxis_title=f"CP2: defensivo (−) ↔ volátil (+)  ({pct.iloc[1]:.1%})")
    st.plotly_chart(estilo(fig, 600), width="stretch")
    nota("Los nombres de los grupos se asignan con reglas sobre su perfil (p. ej., el grupo con mayor dividendo "
         "es 'Defensivas de dividendo'), no por el número que K-Means les da, que es arbitrario. Algunos grupos "
         "se superponen en este plano porque se separan en CP3 y CP4.")

    st.subheader("¿Qué caracteriza a cada grupo? (variables originales)")
    resumen = base.groupby("Grupo")[list(cols)].median()
    resumen.insert(0, "Empresas", base["Grupo"].value_counts())
    resumen["Cap. mediana (mil M USD)"] = base.groupby("Grupo")["Market Cap"].median() / 1e9
    fmt = {c: "{:.3f}" for c in cols}
    fmt.update({"tamano": "{:.2f}", "Cap. mediana (mil M USD)": "{:,.0f}"})
    for c in ["rend_utilidades", "rend_dividendo", "margen_ebitda", "posicion_52s", "rango_52s"]:
        if c in cols:
            fmt[c] = "{:.1%}"
    fmt_legible = {ETIQ.get(c, c): f for c, f in fmt.items()}
    st.dataframe(resumen.rename(columns=ETIQ).style.format(fmt_legible), width="stretch")
    st.caption("Medianas de cada variable dentro del grupo (la mediana es menos sensible a valores extremos que la media).")

    fig = px.imshow(perfil_z.loc[orden_grupos].rename(columns=ETIQ).T, text_auto=".2f", color_continuous_scale=DIVERGENTE,
                    zmin=-2, zmax=2, aspect="auto",
                    title="Perfil de cada grupo en desviaciones estándar frente al promedio del índice")
    fig.update_layout(xaxis_title="", yaxis_title="", coloraxis_colorbar_title="z", xaxis_tickangle=0)
    st.plotly_chart(estilo(fig, 470), width="stretch")
    nota("Rojo = el grupo está por encima del promedio de las empresas analizadas; azul = por debajo. "
         "Por ejemplo, un 2 en dividendo significa que la empresa mediana del grupo paga un dividendo "
         "dos desviaciones estándar por encima del promedio.")

    st.subheader("Empresas más grandes de cada grupo")
    cols_g = st.columns(min(len(orden_grupos), 5))
    for i, g in enumerate(orden_grupos):
        top = base[base["Grupo"] == g].nlargest(8, "Market Cap")
        with cols_g[i % len(cols_g)]:
            st.markdown(f"**{g}** ({(base['Grupo'] == g).sum()})")
            st.markdown("<br>".join(f"{r.Symbol} · {r.Name[:24]}" for r in top.itertuples()), unsafe_allow_html=True)

    st.subheader("¿Los grupos son los sectores con otro nombre?")
    tabla = pd.crosstab(base["Sector"], base["Grupo"])
    fig = px.imshow(tabla, text_auto=True, color_continuous_scale=SECUENCIAL, aspect="auto",
                    title="Número de empresas por sector GICS y grupo")
    fig.update_layout(xaxis_title="", yaxis_title="", coloraxis_colorbar_title="Empresas", xaxis_tickangle=0)
    st.plotly_chart(estilo(fig, 500), width="stretch")
    cv = v_de_cramer(base["Sector"], base["Grupo"])
    nota(f"V de Cramér entre sector y grupo: <b>{cv:.2f}</b> (0 = independientes, 1 = el sector determina el "
         "grupo). Hay una asociación moderada: algunos sectores se concentran en un grupo, pero la mayoría se "
         "reparte entre varios. El sector no basta para describir el perfil financiero de una empresa.")

    st.subheader("Calidad de la asignación: silueta por empresa")
    orden = base.sort_values(["Grupo", "Silueta"], ascending=[True, False]).reset_index(drop=True)
    fig = px.bar(orden, x=orden.index, y="Silueta", color="Grupo", color_discrete_map=color_grupo,
                 category_orders={"Grupo": orden_grupos}, hover_name="Name",
                 hover_data={"Symbol": True, "Silueta": ":.3f"},
                 title="Coeficiente de silueta de cada empresa, ordenado dentro de su grupo")
    fig.update_traces(marker_line_width=0)
    fig.add_hline(y=sil.mean(), line_dash="dash", line_color="#52514e",
                  annotation_text=f"Promedio {sil.mean():.3f}")
    fig.update_layout(xaxis_title="Empresas", yaxis_title="Silueta", bargap=0, xaxis_showticklabels=False)
    st.plotly_chart(estilo(fig, 420), width="stretch")
    porg = base.groupby("Grupo")["Silueta"].agg(["mean", lambda s: (s < 0).mean()])
    porg.columns = ["Silueta media", "% con silueta negativa"]
    st.dataframe(porg.style.format({"Silueta media": "{:.3f}", "% con silueta negativa": "{:.1%}"}),
                 width="stretch")
    st.caption("Una silueta negativa indica que la empresa está, en promedio, más cerca de otro grupo que del suyo: "
               "son los casos difíciles de clasificar.")

# ================================================================= 5. BUSCAR
elif pagina == "5. Buscar una empresa":
    st.title("Buscar una empresa")
    st.markdown("Escoge una empresa para ver su grupo, cómo se compara con su grupo y cuáles son sus "
                "empresas más parecidas en el espacio de componentes.")
    opciones = base.sort_values("Name").apply(lambda r: f"{r.Name} ({r.Symbol})", axis=1).tolist()
    defecto = next((i for i, o in enumerate(opciones) if "(AAPL)" in o), 0)
    elegida = st.selectbox("Empresa", opciones, index=defecto)
    sym = elegida.rsplit("(", 1)[1].rstrip(")")
    i = base.index[base["Symbol"] == sym][0]
    fila = base.loc[i]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grupo", fila["Grupo"])
    c2.metric("Sector GICS", fila["Sector"])
    c3.metric("Capitalización", f"{fila['Market Cap'] / 1e9:,.0f} mil M USD")
    c4.metric("Silueta", f"{fila['Silueta']:.3f}",
              help="Qué tan bien encaja en su grupo. Negativa = está más cerca de otro grupo.")

    z = pd.DataFrame(X, columns=list(cols))
    comp = pd.DataFrame({
        "Empresa": z.loc[i],
        "Mediana del grupo": z[base["Grupo"] == fila["Grupo"]].median(),
    }).rename(index=ETIQ)
    fig = go.Figure()
    fig.add_bar(y=comp.index, x=comp["Empresa"], orientation="h", name=fila["Name"], marker_color=CATEGORICA[0])
    fig.add_bar(y=comp.index, x=comp["Mediana del grupo"], orientation="h", name=f"Mediana: {fila['Grupo']}",
                marker_color=CATEGORICA[1])
    fig.update_layout(barmode="group", title=f"{fila['Name']} frente a su grupo (valores estandarizados)",
                      xaxis_title="Desviaciones estándar frente al promedio del índice", yaxis_title="",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(estilo(fig, 470), width="stretch")

    st.subheader("Empresas más parecidas (vecinos en el espacio de componentes)")
    Zc = puntajes.iloc[:, :n_comp].to_numpy()
    dist = np.linalg.norm(Zc - Zc[i], axis=1)
    vec = base.assign(Distancia=dist).drop(index=i).nsmallest(10, "Distancia")
    st.dataframe(vec[["Symbol", "Name", "Sector", "Grupo", "Distancia"]].style.format({"Distancia": "{:.2f}"}),
                 hide_index=True, width="stretch")
    st.caption("Útil para escoger comparables (peers) en una valoración por múltiplos: los vecinos pueden ser de otro sector.")

# ================================================================= 6. CONCLUSIONES
else:
    st.title("Discusión y conclusiones")
    # Las conclusiones se calculan siempre con la configuración de referencia, sin importar la barra lateral.
    REF = dict(excluir_fin=True, winsor=0.01, cols=tuple(VARIABLES), n_comp=4, k=5)
    b_ref, _ = datos(REF["excluir_fin"], REF["winsor"])
    _, var_ref, car_ref, _ = acp_cache(REF["excluir_fin"], REF["winsor"], REF["cols"])
    g_ref, s_ref, pz_ref, est_ref = segmentar(*REF.values())
    b_ref = b_ref.assign(Grupo=g_ref, Silueta=s_ref)
    ev_ref = k_cache(REF["excluir_fin"], REF["winsor"], REF["cols"], REF["n_comp"])
    cv_ref = v_de_cramer(b_ref["Sector"], b_ref["Grupo"])
    n_g = b_ref["Grupo"].value_counts()

    st.info("Esta sección usa la configuración de referencia: Financials excluido, winsorización al 1 %, "
            "las 8 variables, 4 componentes y K = 5. Los parámetros de la barra lateral no la modifican.")

    st.subheader("1. ¿Qué dimensiones diferencian a las empresas?")
    st.markdown(f"""
Cuatro componentes con valor propio mayor que 1 conservan el **{var_ref['Acumulada'].iloc[3]:.1%}** de la variación
de las ocho razones; los dos primeros, solo el **{var_ref['Acumulada'].iloc[1]:.1%}**. Que haga falta la mitad de las
dimensiones para llegar a tres cuartos de la información indica que **el perfil financiero de una empresa no se reduce
a un solo número**.

- El primer componente ({var_ref['Varianza explicada'].iloc[0]:.1%}) reproduce la distinción clásica de los factores de
  Fama y French entre **crecimiento y valor**: tamaño, P/S y margen altos frente a libro/precio alto. El mercado paga
  múltiplos altos por empresas grandes y rentables.
- El segundo ({var_ref['Varianza explicada'].iloc[1]:.1%}) separa las acciones **volátiles que han caído desde sus
  máximos** de las **defensivas que pagan dividendo**: se parece al factor de baja volatilidad.
- El rendimiento de utilidades forma casi por sí solo el cuarto componente: en esta muestra las ganancias contables
  aportan información distinta a la de los múltiplos de ventas y patrimonio.
""")

    st.subheader("2. ¿Qué grupos aparecen?")
    descr = {
        "Calidad cerca de máximos": "Múltiplos altos, volatilidad baja y precio cerca del máximo del año. Es el grupo "
                                    "más grande e incluye a Apple, Microsoft, Amazon, Eli Lilly o Walmart.",
        "Crecimiento volátil": "Muy caras frente a sus ventas, casi sin dividendo, con el rango de 52 semanas más amplio "
                               "y lejos de sus máximos: NVIDIA, Alphabet, Meta, Broadcom, Tesla, AMD. Concentra buena "
                               "parte de tecnología.",
        "Valor de bajo margen": "Múltiplo P/S bajo, margen EBITDA bajo y libro/precio alto: refinadoras, aseguradoras de "
                                "salud, Comcast y contratistas de defensa (CVS, Valero, Elevance, Northrop).",
        "Defensivas de dividendo": "El mayor dividendo y margen, la menor volatilidad: casi todas las utilities y REITs, "
                                   "telecomunicaciones (Verizon, AT&T) y petroleras integradas.",
        "Con pérdidas contables": "Pocas empresas con utilidad negativa en los últimos doce meses (Ford, Kraft Heinz, "
                                  "Paramount...). Se separan por el cuarto componente, no por el sector.",
    }
    filas = []
    for g in sorted(n_g.index):
        filas.append({"Grupo": g, "Empresas": int(n_g[g]),
                      "Silueta media": b_ref.loc[b_ref["Grupo"] == g, "Silueta"].mean(),
                      "Descripción": descr.get(g, "")})
    st.dataframe(pd.DataFrame(filas).style.format({"Silueta media": "{:.2f}"}), hide_index=True,
                 width="stretch")
    st.markdown(f"""
K = 5 tiene la silueta más alta entre 2 y 10 grupos con 4 componentes (**{s_ref.mean():.3f}**), y la partición
es estable: con semillas distintas se recupera en buena medida (ARI medio **{est_ref:.2f}**). Hacer K-Means sobre las
ocho variables estandarizadas, en lugar de sobre los componentes, da una partición casi idéntica (ARI ≈ 0,9), así que
el resultado no depende de esa decisión.
""")

    st.subheader("3. ¿Son grupos claramente distinguibles?")
    st.markdown(f"""
**No del todo, y esto también es un hallazgo.** Una silueta promedio cercana a 0,27 indica que los grupos existen pero
se tocan. Las empresas forman más bien un **continuo** de estilos con zonas más densas que racimos separados por vacíos.
Los grupos más nítidos son *Con pérdidas contables* y *Calidad cerca de máximos*; los más difusos son *Valor de bajo
margen* y *Crecimiento volátil*, con siluetas cercanas a 0,17–0,18. Las empresas con silueta negativa
({(s_ref < 0).mean():.1%} del total) están en la frontera y deben leerse con cuidado.
""")

    st.subheader("4. ¿Los grupos son los sectores?")
    st.markdown(f"""
Solo en parte (V de Cramér = **{cv_ref:.2f}**). Casi todas las utilities (25 de 31) y dos tercios de Real Estate (21 de 31) caen en *Defensivas de
dividendo*, como cabría esperar de negocios regulados con flujos estables. Pero **tecnología se parte en dos**:
la mitad está en *Crecimiento volátil* (semiconductores e IA) y otra parte en *Calidad cerca de máximos*
(software y hardware maduros). Industriales, salud y consumo se reparten entre tres grupos. Para un inversionista
esto significa que **diversificar por sector no garantiza diversificar por perfil**: una cartera con una acción de
cada sector puede quedar cargada al mismo estilo.
""")

    st.subheader("5. Limitaciones")
    st.markdown("""
- **Una sola fecha.** La posición y la amplitud del rango de 52 semanas, y todos los múltiplos, dependen del precio de
  un día. El grupo *Crecimiento volátil* está definido en parte por haber caído desde sus máximos del año (posición mediana ≈ 0,25), algo coyuntural; en otro
  momento del ciclo el mapa sería distinto. Repetir el análisis en varias fechas mostraría qué grupos son estructurales.
- **Calidad de la fuente.** Yahoo Finance no está auditado y mezcla fechas de corte contable; los valores extremos se
  recortaron, lo que cambia los resultados (se puede comprobar moviendo la winsorización en la barra lateral).
- **Sesgo de selección.** Se excluyeron bancos y aseguradoras, y solo se estudian empresas grandes de EE. UU. Las
  conclusiones no se extienden a empresas pequeñas ni a mercados como el colombiano.
- **K-Means supone grupos esféricos y de tamaño parecido.** Con estructura débil, métodos como clústeres jerárquicos o
  mezclas gaussianas podrían dar fronteras distintas. Los grupos son una herramienta descriptiva, no una
  clasificación definitiva ni una recomendación de inversión.
""")
