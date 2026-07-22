"""Módulo de ingesta de datos."""

import pandas as pd


def cargar_csv(ruta: str) -> pd.DataFrame:
    """
    Carga un archivo CSV y verifica que tenga al menos una fila.

    Args:
        ruta: Ruta del archivo CSV.

    Returns:
        DataFrame con los datos.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío.
    """
    df = pd.read_csv(ruta)

    if df.empty:
        raise ValueError("El archivo no contiene filas")

    return df
