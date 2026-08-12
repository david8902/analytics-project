"""Módulo de ingesta de datos."""

from pathlib import Path

import pandas as pd

# Columnas de fecha que deben parsearse como datetime64 en cada tabla de Olist.
_COLUMNAS_FECHA = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}

# Nombre del archivo CSV correspondiente a cada tabla dentro de data_dir.
_ARCHIVOS_OLIST = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
}


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


def cargar_olist(data_dir: str) -> dict[str, pd.DataFrame]:
    """Carga las 5 tablas principales de Olist desde data_dir.

    Registra con print() la forma (filas × columnas) de cada tabla al cargarla.
    Las columnas de fecha se cargan como datetime64, no como object.

    Args:
        data_dir: ruta al directorio que contiene los archivos CSV de Olist.

    Returns:
        Dict con keys 'orders', 'items', 'customers', 'payments', 'reviews',
        cada uno mapeado a su DataFrame correspondiente.

    Raises:
        FileNotFoundError: si alguno de los 5 archivos no existe en data_dir.
    """
    directorio = Path(data_dir)
    tablas: dict[str, pd.DataFrame] = {}

    for nombre_tabla, nombre_archivo in _ARCHIVOS_OLIST.items():
        ruta = directorio / nombre_archivo

        if not ruta.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

        columnas_fecha = _COLUMNAS_FECHA.get(nombre_tabla)
        df = pd.read_csv(ruta, parse_dates=columnas_fecha)

        print(f"[CARGA] {nombre_tabla}: {df.shape[0]} filas × {df.shape[1]} columnas")
        tablas[nombre_tabla] = df

    return tablas


def join_verificado(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: str | list[str],
    how: str = "left",
    nombre: str = "join",
) -> pd.DataFrame:
    """Realiza un merge y verifica que el resultado no multiplique filas.

    Un join que multiplica filas indica que la tabla derecha tiene duplicados
    en la columna clave — error silencioso sin esta verificación.

    Args:
        df_left: DataFrame izquierdo (el que define el número de filas esperado).
        df_right: DataFrame derecho.
        on: columna(s) clave del join.
        how: tipo de join ('left', 'inner', 'outer', 'right').
        nombre: nombre descriptivo para el mensaje de error.

    Returns:
        DataFrame resultante del merge.

    Raises:
        AssertionError: si el resultado tiene más filas que df_left.
    """
    filas_esperadas = len(df_left)
    resultado = df_left.merge(df_right, on=on, how=how)
    filas_obtenidas = len(resultado)

    assert filas_obtenidas <= filas_esperadas, (
        f"{nombre}: el join produjo más filas de las esperadas "
        f"({filas_obtenidas} filas vs. {filas_esperadas} filas de entrada). "
        "Esto indica que la tabla derecha tiene claves duplicadas."
    )

    return resultado


def construir_dataset_base(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combina las tablas de Olist en un único DataFrame analítico.

    Estrategia de joins:
    - orders × customers → left join on customer_id
    - + payments_agg → left join on order_id (payments debe agregarse primero)
    - + items_agg → left join on order_id (items debe agregarse primero)

    Agrega payments antes del join: total_pago (sum) y n_cuotas (max) por order_id.
    Agrega items antes del join: n_items (count) y ticket_total (sum de price) por
    order_id.

    Args:
        tablas: dict retornado por cargar_olist().

    Returns:
        DataFrame con una fila por pedido (99,441 filas si los datos son completos).
    """
    orders = tablas["orders"]
    customers = tablas["customers"]
    payments = tablas["payments"]
    items = tablas["items"]

    payments_agg = (
        payments.groupby("order_id")
        .agg(
            total_pago=("payment_value", "sum"),
            n_cuotas=("payment_installments", "max"),
        )
        .reset_index()
    )

    items_agg = (
        items.groupby("order_id")
        .agg(n_items=("order_item_id", "count"), ticket_total=("price", "sum"))
        .reset_index()
    )

    dataset = join_verificado(
        orders, customers, on="customer_id", how="left", nombre="orders_customers"
    )
    dataset = join_verificado(
        dataset, payments_agg, on="order_id", how="left", nombre="orders_payments"
    )
    dataset = join_verificado(
        dataset, items_agg, on="order_id", how="left", nombre="orders_items"
    )

    return dataset
