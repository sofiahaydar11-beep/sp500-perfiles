"""ACP y K-Means. Funciones puras que la app llama con los parámetros que elige el usuario."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

SEMILLA = 42


def estandarizar(base: pd.DataFrame, cols: list[str]):
    escalador = StandardScaler()
    X = escalador.fit_transform(base[cols])
    return X, escalador


def ajustar_acp(X: np.ndarray, cols: list[str]):
    """ACP con todos los componentes, para mostrar varianza, sedimentación y cargas."""
    acp = PCA(random_state=SEMILLA).fit(X)
    n = len(cols)
    nombres = [f"CP{i + 1}" for i in range(n)]
    varianza = pd.DataFrame({
        "Componente": nombres,
        "Valor propio": acp.explained_variance_,
        "Varianza explicada": acp.explained_variance_ratio_,
        "Acumulada": np.cumsum(acp.explained_variance_ratio_),
    })
    cargas = pd.DataFrame(acp.components_.T, index=cols, columns=nombres)
    puntajes = pd.DataFrame(acp.transform(X), columns=nombres)
    return acp, varianza, cargas, puntajes


def sugerir_componentes(varianza: pd.DataFrame, umbral: float = 0.75) -> dict:
    """Dos criterios: Kaiser (valor propio > 1) y varianza acumulada mínima."""
    kaiser = int((varianza["Valor propio"] > 1).sum())
    acumulado = int((varianza["Acumulada"] < umbral).sum() + 1)
    return {"kaiser": kaiser, "acumulado": acumulado}


def evaluar_k(Z: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(Z)
        filas.append({"K": k, "Inercia": km.inertia_, "Silueta": silhouette_score(Z, km.labels_)})
    return pd.DataFrame(filas)


def ajustar_kmeans(Z: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA).fit(Z)
    return km, silhouette_samples(Z, km.labels_)


def estabilidad(Z: np.ndarray, k: int, etiquetas: np.ndarray, repeticiones: int = 20) -> float:
    """ARI medio entre la solución final y soluciones con una sola inicialización y semillas distintas.
    1 = la partición no depende del punto de partida; cerca de 0 = muy inestable."""
    aris = [adjusted_rand_score(etiquetas, KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(Z))
            for s in range(repeticiones)]
    return float(np.mean(aris))


def v_de_cramer(a: pd.Series, b: pd.Series) -> float:
    """Asociación entre dos variables categóricas (0 = ninguna, 1 = una determina la otra)."""
    tabla = pd.crosstab(a, b).to_numpy()
    n = tabla.sum()
    esperado = tabla.sum(1, keepdims=True) * tabla.sum(0, keepdims=True) / n
    chi2 = ((tabla - esperado) ** 2 / esperado).sum()
    r, c = tabla.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def nombrar_grupos(perfil_z: pd.DataFrame) -> dict:
    """Asigna nombres descriptivos a partir del perfil estandarizado (mediana en z) de cada grupo.

    Las reglas se aplican en orden y cada grupo recibe un solo nombre; las que no encajan
    quedan como 'Grupo X'. Así el nombre depende de las características, nunca del número
    que K-Means le asigna al clúster.
    """
    restantes = list(perfil_z.index)
    nombres = {}
    reglas = [
        ("Con pérdidas contables", "rend_utilidades", "min", -1.5),
        ("Defensivas de dividendo", "rend_dividendo", "max", 0.6),
        ("Crecimiento volátil", "rango_52s", "max", 0.6),
        ("Valor de bajo margen", "log_precio_ventas", "min", -0.4),
        ("Calidad cerca de máximos", "posicion_52s", "max", 0.3),
    ]
    for nombre, var, sentido, umbral in reglas:
        if not restantes or var not in perfil_z.columns:
            continue
        sub = perfil_z.loc[restantes, var]
        g = sub.idxmin() if sentido == "min" else sub.idxmax()
        valor = sub[g]
        if (sentido == "min" and valor <= umbral) or (sentido == "max" and valor >= umbral):
            nombres[g] = nombre
            restantes.remove(g)
    letras = iter("ABCDEFGHIJ")
    for g in restantes:
        nombres[g] = f"Grupo {next(letras)}"
    return nombres
