import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
import time

# --- Configuración de la Página ---
st.set_page_config(
    page_title="IMDb Advanced Analytics & AI",
    page_icon="🎬",
    layout="wide"
)

# --- Carga y Limpieza de Datos (Cached) ---
@st.cache_data
def load_data():
    # URL ESTABLE (Fuente original del dataset)
    url = "https://raw.githubusercontent.com/harshitshankhdhar/IMDB-Dataset-Analysis/master/imdb_top_1000.csv"
    
    try:
        df = pd.read_csv(url)
        
        # 0. Limpieza de nombres de columnas (Crucial para evitar errores de espacios)
        df.columns = df.columns.str.strip()
        
        # 1. Limpieza de 'Gross'
        # Convertimos a string, quitamos comas y convertimos a número
        if 'Gross' in df.columns:
            df['Gross'] = df['Gross'].astype(str).str.replace(',', '').replace('nan', '0')
            df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce').fillna(0)
        
        # 2. Limpieza de 'Released_Year'
        if 'Released_Year' in df.columns:
            df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
            df = df.dropna(subset=['Released_Year']) 
            df['Released_Year'] = df['Released_Year'].astype(int)
            
        return df
    except Exception as e:
        st.error(f"Error cargando los datos desde la URL de respaldo: {e}")
        return pd.DataFrame()

df = load_data()

# --- VALIDACIÓN DE SEGURIDAD ---
if df.empty:
    st.error("❌ No se pudieron cargar los datos. Verifica tu conexión a internet.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Inserta tu clave aquí...")
    
    st.divider()
    
    # Filtros
    # Usamos sorted y unique para llenar los selectores
    directors = sorted(df['Director'].dropna().unique())
    selected_director = st.selectbox("Selecciona Director", directors)
    
    genres = sorted(df['Genre'].dropna().unique())
    selected_genres = st.multiselect("Filtrar por Género (Opcional)", genres)

# --- Filtrado de Datos ---
filtered_df = df[df['Director'] == selected_director]

if selected_genres:
    # Filtro flexible para géneros múltiples
    pattern = '|'.join(selected_genres)
    filtered_df = filtered_df[filtered_df['Genre'].str.contains(pattern, case=False, na=False)]

# --- Main Dashboard ---
st.title(f"🎬 Dashboard: {selected_director}")

if filtered_df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
else:
    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_gross = filtered_df['Gross'].sum()
    avg_rating = filtered_df['IMDB_Rating'].mean()
    avg_meta = filtered_df['Meta_score'].mean()
    
    # Obtener película más taquillera
    if not filtered_df['Gross'].empty and filtered_df['Gross'].max() > 0:
        top_movie_idx = filtered_df['Gross'].idxmax()
        top_movie = filtered_df.loc[top_movie_idx, 'Series_Title']
    else:
        top_movie = "N/A"

    col1.metric("Total Recaudación", f"${total_gross:,.0f}")
    col2.metric("Rating Promedio (IMDb)", f"{avg_rating:.1f}")
    col3.metric("Meta Score Promedio", f"{avg_meta:.1f}")
    col4.metric("Top Taquilla", top_movie)

    st.divider()

    # --- Gráficos (Plotly) ---
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.subheader("💰 Top 10 Recaudación")
        df_top_gross = filtered_df.sort_values(by='Gross', ascending=False).head(10)
        
        fig_bar = px.bar(
            df_top_gross, 
            x='Series_Title', 
            y='Gross',
            color='Gross',
            color_continuous_scale='Greens'
        )
        fig_bar.update_layout(xaxis_title="Película", yaxis_title="Recaudación ($)", autosize=True)
        st.plotly_chart(fig_bar) 

    with c_chart2:
        st.subheader("⭐ Rating vs Recaudación")
        fig_scatter = px.scatter(
            filtered_df,
            x='IMDB_Rating',
            y='Gross',
            hover_data=['Series_Title', 'Released_Year'],
            color='IMDB_Rating',
            color_continuous_scale='Bluered'
        )
        fig_scatter.update_layout(xaxis_title="IMDb Rating", yaxis_title="Recaudación ($)", autosize=True)
        st.plotly_chart(fig_scatter)

    # --- Integración Gemini AI (Chat) ---
    st.divider()
    st.subheader("🤖 AI Cine-Analista")
    st.info("Pregunta sobre la filmografía del director filtrado.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ej: ¿Cuál es su película más aclamada y por qué?"):
        
        if not api_key:
            st.error("⚠️ Por favor ingresa tu Gemini API Key en el sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Smart Context Logic
        context_cols = ['Series_Title', 'Released_Year', 'IMDB_Rating', 'Gross', 'Director']
        # Validar que columnas existen antes de filtrar
        valid_cols = [c for c in context_cols if c in filtered_df.columns]
        context_df = filtered_df[valid_cols].copy()

        warning_msg = ""
        # Optimización: si hay muchas filas, enviamos solo el Top 50 por rating
        if len(context_df) > 50:
            context_df = context_df.sort_values(by='IMDB_Rating', ascending=False).head(50)
            warning_msg = "(Nota: Se envía solo el Top 50 por rating para optimizar el análisis)."
        
        data_context = context_df.to_csv(index=False)

        system_instruction = f"""
        Eres un experto analista de cine. Tienes acceso a los datos de películas del director {selected_director}.
        Datos disponibles (CSV):
        {data_context}
        {warning_msg}
        
        Responde a la pregunta del usuario basándote estrictamente en estos datos.
        """

        with st.chat_message("assistant"):
            try:
                status_container = st.status("🎬 Analizando filmografía...", expanded=True)
                
                client = genai.Client(api_key=api_key)
                
                response_stream = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=[system_instruction, prompt]
                )
                
                status_container.update(label="💡 Respuesta generada", state="complete", expanded=False)
                
                def stream_generator():
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text

                full_response = st.write_stream(stream_generator())
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                status_container.update(label="❌ Error", state="error")
                st.error(f"Error conectando con Gemini: {str(e)}")
