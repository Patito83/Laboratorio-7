# Laboratorio 7

## Análisis de sentimientos

Utilpy
## Desarrollo de un ETL 
### Proyecto ETL + Streamlit - Sensores (Túnel Carpiano)

Mediante conocimientos de estructuras ETl y streamlit, se realizo un analisis completo de resultados de sensores para el Tunel Carpiano. Mediante ETL esos resulados se leyeron y se implementaron en una grafica para entender los resultados del tuneol Carpiano.

#### Pasos ETL

1. Primero creamos la carpeta donde estara todo nuestro proyecto (  `mkdir proyecto_STC` ) y luego creamos nuestro archivo de ETL `etl.py`

```
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = "/mnt/data/BD_SENSORES.xlsx"
OUTPUT_WIDE = "dataset_wide.csv"
OUTPUT_LONG = "dataset_long.csv"

def to_float_series(s):
    if s.dtype == object:
        s2 = s.astype(str).str.replace("V", "", regex=False).str.replace("v", "", regex=False)
        s2 = s2.str.strip()
        return pd.to_numeric(s2, errors="coerce")
    else:
        return pd.to_numeric(s, errors="coerce")

def rms_series(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr**2)))

def process_sheet(df, sheet_name):
    df = df.copy()
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    if df.shape[0] == 0 or df.shape[1] == 0:
        return None, []
    cols = list(df.columns)
    usuario_col = None
    for c in cols:
        if str(c).strip().lower() == "usuario":
            usuario_col = c
            break
    sensor_cols = [c for c in cols if c != usuario_col]
    for c in sensor_cols:
        df[c] = to_float_series(df[c])
    df = df.dropna(axis=1, how="all")
    cols = list(df.columns)
    if usuario_col and usuario_col not in cols:
        usuario_col = None
    sensor_cols = [c for c in cols if c != usuario_col]
    if len(sensor_cols) == 0:
        return None, []
    wide_rows = []
    long_rows = []
    for idx, row in df.iterrows():
        usuario_val = row[usuario_col] if usuario_col else f"{sheet_name}_row{idx}"
        values = row[sensor_cols].astype(float)
        valid_vals = values.dropna().values
        stats = {
            "sheet": sheet_name,
            "row_index": int(idx),
            "Usuario": usuario_val,
            "n_valid": int(np.sum(~np.isnan(values))),
            "mean": float(np.nanmean(valid_vals)) if valid_vals.size > 0 else np.nan,
            "std": float(np.nanstd(valid_vals, ddof=1)) if valid_vals.size > 1 else np.nan,
            "rms": rms_series(valid_vals),
            "amax": float(np.nanmax(valid_vals)) if valid_vals.size > 0 else np.nan,
            "amin": float(np.nanmin(valid_vals)) if valid_vals.size > 0 else np.nan,
        }
        wide_row = {"sheet": sheet_name, "row_index": int(idx), "Usuario": usuario_val}
        for c in sensor_cols:
            wide_col_name = f"C{c}" if isinstance(c, (int, float)) else str(c)
            wide_row[wide_col_name] = row[c] if not pd.isna(row[c]) else np.nan
            long_rows.append({
                "sheet": sheet_name,
                "row_index": int(idx),
                "Usuario": usuario_val,
                "canal": wide_col_name,
                "value": float(row[c]) if not pd.isna(row[c]) else np.nan
            })
        wide_row.update(stats)
        wide_rows.append(wide_row)
    wide_df = pd.DataFrame(wide_rows)
    return wide_df, long_rows

def main():
    xls = pd.ExcelFile(INPUT_PATH)
    all_wide = []
    all_long = []
    for sheet in xls.sheet_names:
        df_raw = pd.read_excel(xls, sheet_name=sheet, header=0)
        wide_df, long_rows = process_sheet(df_raw, sheet)
        if wide_df is None:
            continue
        all_wide.append(wide_df)
        all_long.extend(long_rows)
    if len(all_wide) == 0:
        return
    df_wide = pd.concat(all_wide, ignore_index=True)
    df_long = pd.DataFrame(all_long)
    df_wide.to_csv(OUTPUT_WIDE, index=False)
    df_long.to_csv(OUTPUT_LONG, index=False)

if __name__ == "__main__":
    main()
```
2. Este codigo lo implementamos en un entorno virtual, entonces creamos un archivo de texto con todas las librerias a descargar.

`` pandas
numpy
streamlit
plotly
openpyxl``

3. Ejecutamos el ETL y las librerias. ``pip install -r requirements.txt`` y ``python etl.py``

#### Pasos Streamlit

Dentro de este apartado la idea es desplegar graficas que interactuen y explique los datos recogidos de los resultados de sensores.

1. Creamos el StreamLit `streamlit_app.py` y agregamos el siguiente codigo

```
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

WIDE_PATH = "dataset_wide.csv"
LONG_PATH = "dataset_long.csv"

st.set_page_config(page_title="Dashboard Sensores - ETL", layout="wide")
st.title("Dashboard - ETL Sensores")

try:
    df_wide = pd.read_csv(WIDE_PATH)
    df_long = pd.read_csv(LONG_PATH)
except Exception as e:
    st.error(f"No se pudo leer los archivos de ETL. Ejecuta `python etl.py` primero.\nError: {e}")
    st.stop()

st.sidebar.header("Filtros")
sheets = sorted(df_wide["sheet"].unique().tolist())
sheet_sel = st.sidebar.selectbox("Seleccionar hoja", sheets)

df_wide_s = df_wide[df_wide["sheet"] == sheet_sel]
df_long_s = df_long[df_long["sheet"] == sheet_sel]

usuarios = df_wide_s["Usuario"].fillna("sin_usuario").unique().tolist()
usuario_sel = st.sidebar.selectbox("Seleccionar Usuario", usuarios)

df_wide_user = df_wide_s[df_wide_s["Usuario"].fillna("sin_usuario") == usuario_sel]
if df_wide_user.empty:
    df_wide_user = df_wide_s.head(1)

row_index = int(df_wide_user["row_index"].iloc[0])

canales = sorted(df_long_s["canal"].unique().tolist())
canal_sel = st.sidebar.selectbox("Seleccionar canal", canales)

st.header(f"Hoja: {sheet_sel} — Usuario: {usuario_sel}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filas", f"{len(df_wide_s)}")
col2.metric("Muestras/canal", f"{df_long_s.groupby('canal').size().median():.0f}")
col3.metric("Canales", f"{len(canales)}")
col4.metric("Muestras válidas", f"{int(df_wide_s['n_valid'].median())}")

st.subheader("Serie del canal seleccionado")
df_series = df_long_s[(df_long_s["canal"] == canal_sel) & (df_long_s["row_index"] == row_index)].copy()

if df_series.empty:
    st.warning("No hay datos para este canal/usuario.")
else:
    fig = px.line(df_series.reset_index(), x=df_series.reset_index().index, y="value",
                  labels={"index": "muestra", "value": "Valor"},
                  title=f"{canal_sel} — fila {row_index}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Histograma")
    fig2 = px.histogram(df_series, x="value", nbins=40, title=f"Histograma {canal_sel}")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Estadísticas")
    mean_v = df_series["value"].mean()
    std_v = df_series["value"].std()
    rms_v = (df_series["value"].dropna()**2).mean()**0.5 if df_series["value"].dropna().size > 0 else float("nan")
    max_v = df_series["value"].max()
    min_v = df_series["value"].min()

    st.write({
        "mean": float(mean_v) if pd.notna(mean_v) else None,
        "std": float(std_v) if pd.notna(std_v) else None,
        "rms": float(rms_v) if pd.notna(rms_v) else None,
        "max": float(max_v) if pd.notna(max_v) else None,
        "min": float(min_v) if pd.notna(min_v) else None
    })

st.subheader("Vista previa (dataset_wide)")
st.dataframe(df_wide_s.head(10))

st.subheader("Descargar datos")
csv = df_long_s.to_csv(index=False)
st.download_button(
    label=f"Descargar dataset_long_{sheet_sel}.csv",
    data=csv,
    file_name=f"dataset_long_{sheet_sel}.csv",
    mime="text/csv"
)
```

2. Luego ejecutamos el streamLit ``streamlit run streamlit_app.py`` y visualizaremos los resultados.

<img width="1327" height="314" alt="image" src="https://github.com/user-attachments/assets/1e5e3451-a2af-4e79-ae66-181b2f5f0877" />


## Exploración Tecnologica 

## MinCiencias de IA

### Detección Temprana de Plagas y Estrés Hídrico

Convocatoria MinCiencias 

Como estudiantes de la USTA proponemos un proyecto que desarrolla una solución de Inteligencia Artificial para el monitoreo agrícola en territorios rurales. El sistema utiliza nodos IoT de bajo costo con sensores ambientales y cámaras de baja resolución para detectar plagas y estrés hídrico en cultivos, enviando alertas tempranas mediante redes. 

#### Problematica Principal

Realizando un análisis dentro de Colombia encontramos que los habitantes de zonas rurales sufren perdidas de producción. Esto por detección tardía de plagas y riego ineficiente por falta de monitoreo continuo. Esto provoca baja productividad y altos costos operativos. La infraestructura de red es limitada, por lo que se requieren soluciones de bajo consumo, economía de datos y procesamiento local.

#### Obgetivos General

Desarrollar una plataforma IoT con IA embebida que permita:

1. Monitorear cultivos en tiempo real.

2. Detectar tempranamente plagas y estrés hídrico.

3. Enviar alertas a los agricultores con baja latencia.

4. Crear un dataset local reutilizable por la comunidad.

#### Flujo de Operación

- Nodo toma datos.

- Ejecuta inferencia local con TinyML.

- Si detecta plaga/estrés → envía alerta por LoRaWAN.

- Backend guarda, clasifica y notifica.

- Dashboard muestra métricas y recomendaciones.

- Datos nuevos alimentan el pipeline de reentrenamiento.
