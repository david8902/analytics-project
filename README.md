# Analytics Project

Proyecto del semestre de Analítica de Datos.

## Instalación

```bash
python -m uv sync

#8.1	¿Qué pasó al ejecutar uv run pytest antes de implementar cargar_csv? ¿Qué error apareció y qué dice ese error sobre cómo funciona pytest?
#RTA: Antes de implementar cargar_csv, los tests fallaban porque la función todavía no estaba desarrollada y contenía raise NotImplementedError. Al ejecutar uv run pytest, pytest ejecutó automáticamente las pruebas y reportó cuáles fallaban, indicando el error y el archivo donde ocurrió. Esto demuestra que pytest verifica automáticamente el comportamiento del código y compara el resultado con lo que esperan las pruebas.
#8.2	¿Por qué pd.read_csv() lanza FileNotFoundError automáticamente? ¿Qué implica eso para la implementación?
#RTA: pd.read_csv() intenta abrir el archivo indicado por la ruta. Si el archivo no existe, Python genera automáticamente la excepción FileNotFoundError. Por esta razón no es necesario escribir código adicional para detectar ese caso; basta con dejar que la excepción se propague. En la implementación solo fue necesario validar que el archivo cargado contuviera al menos una fila de datos y, en caso contrario, lanzar un ValueError.
#8.3	El CI tardó ~60 segundos en correr en GitHub, pero uv run pytest en la máquina local tardó menos de un segundo. ¿A qué se debe esa diferencia? ¿Podría optimizarse?
#RTA: La diferencia se debe a que GitHub Actions crea un entorno nuevo en cada ejecución. Antes de correr las pruebas debe descargar el repositorio, instalar Python, instalar uv, resolver e instalar las dependencias y luego ejecutar los tests. En mi computador ese entorno ya estaba preparado, por lo que las pruebas se ejecutaron mucho más rápido. El tiempo del CI puede reducirse utilizando caché para las dependencias y evitando reinstalar paquetes que no hayan cambiado.


#RESPUESTAS A PREGUNTAS GUIA 
#¿pd.read_csv() lanza FileNotFoundError automáticamente si el archivo no existe, o se necesita manejarlo explícitamente? 
#RTA: pd.read_csv() lanza automáticamente una excepción FileNotFoundError cuando el archivo especificado no existe. Si deseas mostrar un mensaje personalizado o evitar que el programa termine, debes capturarla con un bloque try-except.

#¿Cómo se distingue un DataFrame vacío (sin filas) de uno con datos? ¿Qué atributo o método de pandas ayuda?
#RTA:El atributo empty del Dataframe es el que ayuda

