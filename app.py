import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types

# --- 1. Configuración de la Página ---
st.set_page_config(
    page_title="Energy Dashboard IA",
    page_icon="⚡",
    layout="wide"
)

# --- 2. Carga y Limpieza de Datos ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

    # Filtrar regiones agregadas
    df = df[df['iso_code'].notna()]

    # Rellenar NAs numéricos con 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Definir columnas
    renovables_cols = ['solar_consumption', 'wind_consumption', 'hydro_consumption']
    fosiles_cols = ['coal_consumption', 'oil_consumption', 'gas_consumption']
    
    # Asegurar que existan
    for col in renovables_cols + fosiles_cols:
        if col not in df.columns:
            df[col] = 0

    df['Total Renovables'] = df[renovables_cols].sum(axis=1)
    df['Total Fósiles'] = df[fosiles_cols].sum(axis=1)

    return df

df = load_data()

# --- 3. Sidebar y Filtros ---
st.sidebar.header("Configuración")

# Seguridad: API Key
api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Inserta tu clave aquí")

# Filtros
if not df.empty:
    paises = sorted(df['country'].unique())
    default_idx = paises.index('Spain') if 'Spain' in paises else 0
    
    selected_country = st.sidebar.selectbox("Selecciona País", paises, index=default_idx)
    
    country_data = df[df['country'] == selected_country]
    min_year = int(country_data['year'].min())
    max_year = int(country_data['year'].max())
    
    selected_year = st.sidebar.slider("Selecciona Año", min_year, max_year, max_year)
else:
    st.stop()

df_country = df[df['country'] == selected_country]
df_year = df_country[df_country['year'] == selected_year]

# --- 4. Interfaz Principal (KPIs) ---
st.title(f"⚡ Dashboard Energético: {selected_country}")

if not df_year.empty:
    solar = df_year['solar_consumption'].values[0]
    wind = df_year['wind_consumption'].values[0]
    fossil = df_year['Total Fósiles'].values[0]
    total_renovables = df_year['Total Renovables'].values[0]
    
    total_mix = total_renovables + fossil
    pct_renovable = (total_renovables / total_mix * 100) if total_mix > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Solar (TWh)", f"{solar:.2f}")
    col2.metric("Eólica (TWh)", f"{wind:.2f}")
    col3.metric("Fósiles (TWh)", f"{fossil:.2f}")
    col4.metric("% Renovables", f"{pct_renovable:.1f}%")
else:
    st.warning("No hay datos para el año seleccionado.")

st.divider()

# --- 5. Visualización (Plotly) ---
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Evolución Histórica")
    fig_line = px.line(
        df_country, 
        x='year', 
        y=['Total Renovables', 'Total Fósiles'],
        labels={'value': 'Consumo (TWh)', 'variable': 'Fuente'},
        title=f"Evolución en {selected_country}"
    )
    # CORRECCIÓN AQUÍ: Usamos un diccionario para la leyenda, no 'legend_position'
    fig_line.update_layout(
        autosize=True, 
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_line)

with col_chart2:
    st.subheader(f"Mix Energético ({selected_year})")
    if not df_year.empty:
        mix_data = {
            'Fuente': ['Solar', 'Eólica', 'Hidro', 'Carbón', 'Petróleo', 'Gas'],
            'Consumo': [
                df_year['solar_consumption'].values
