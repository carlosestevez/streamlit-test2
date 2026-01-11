import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="IMDb Advanced Analytics & AI",
    page_icon="🎬",
    layout="wide"
)

# --- Carga y Limpieza de Datos ---
@st.cache_data
def load_data():
    # 1. Intentar cargar desde archivo local
    local_file = "imdb_top_1000.csv"
    if os.path.exists(local_file):
        try:
            df = pd.read_csv(local_file)
            return clean_dataframe(df)
        except Exception:
            pass

    # 2. Intentar cargar desde URLs espejo
    urls = [
        "https://raw.githubusercontent.com/JaviRute/top_1000_movies-data_science_project/master/imdb_top_1000.csv",
        "https://raw.githubusercontent.com/Elliott-dev/Top-1000-IMDB-Rated-Movies-Analysis/master/imdb_top_1000.csv",
        "https://raw.githubusercontent.com/krishna-koly/IMDB_TOP_1000/main/imdb_top_1000.csv"
    ]
    
    for url in urls:
        try:
            df = pd.read_csv(url, on_bad_lines='skip')
            if 'Director' in df.columns and 'Gross' in df.columns:
                return clean_dataframe(df)
        except Exception:
            continue 
            
    return pd.DataFrame()

def clean_dataframe(df):
    try:
        df.columns = df.columns.str.strip()
        
        # Limpieza Gross
        if 'Gross' in df.columns:
            df['Gross'] = df['Gross'].astype(str).str.replace(',', '').replace('nan', '0')
            df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce').fillna(0)
            
        # Limpieza Año
        if 'Released_Year' in df.columns:
            df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
            df = df.dropna(subset=['Released_Year'])
            df['Released_Year'] = df['Released_Year'].astype(int)
            
        return df
    except Exception as e:
        return pd.DataFrame()

with st.spinner("Cargando datos..."):
    df = load_data()

if df.empty:
    st.error("❌ No se pudo descargar el dataset. Revisa tu conexión.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Inserta tu clave aquí...")
    st.divider()
    
    # 1. Filtro Director (con opción 'Todos')
    directors = sorted(df['Director'].dropna().unique().tolist())
    directors.insert(0, "Todos")  # Añadimos la opción 'Todos' al principio
    selected_director = st.selectbox("Selecciona Director", directors)
    
    # 2. Filtro Género
    # Extraemos todos los géneros únicos separando por coma
    all_genres_raw = df['Genre'].dropna().astype(str).tolist()
    unique_genres = set()
    for g_str in all_genres_raw:
        parts = [p.strip() for p in g_str.split(',')]
        unique_genres.update(parts)
    
    genres = sorted(list(unique_genres))
    selected_genres = st.multiselect("Filtrar por Género", genres)

# --- Lógica de Filtrado Principal ---

# 1. Filtro por Director
if selected_director == "Todos":
    filtered_df = df.copy()
    director_title = "Análisis Global"
else:
    filtered_df = df[df['Director'] == selected_director]
    director_title = f"Director: {selected_director}"

# 2. Filtro por Género (aplica sobre lo anterior)
if selected_genres:
    # Creamos patrón regex para buscar cualquiera de los géneros seleccionados
    pattern = '|'.join(selected_genres)
    filtered_df = filtered_df[filtered_df['Genre'].astype(str).str.contains(pattern, case=False, na=False)]

# --- Dashboard ---
st.title(f"🎬 {director_title}")

if filtered_df.empty:
    st.warning("⚠️ No se encontraron películas con los filtros seleccionados.")
else:
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_gross = filtered_df['Gross'].sum()
    avg_rating = filtered_df['IMDB_Rating'].mean()
    meta_score = filtered_df['Meta_score'].mean() if 'Meta_score' in filtered_df.columns else 0
    
    top_movie = "N/A"
    if not filtered_df['Gross'].empty and filtered_df['Gross'].max() > 0:
        top_movie = filtered_df.loc[filtered_df['Gross'].idxmax(), 'Series_Title']

    col1.metric("Total Recaudación", f"${total_gross:,.0f}")
    col2.metric("Rating IMDb", f"{avg_rating:.1f}")
    col3.metric("Meta Score", f"{meta_score:.1f}")
    col4.metric("Top Taquilla", top_movie)

    st.divider()

    # Gráficos
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 Top Recaudación")
        # Mostramos Top 10 de la selección actual
        df_chart = filtered_df.sort_values('Gross', ascending=False).head(10)
        
        fig = px.bar(df_chart, x='Series_Title', y='Gross', color='Gross', color_continuous_scale='Greens')
        fig.update_layout(autosize=True, xaxis_title="", yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("⭐ Rating vs Recaudación")
        fig2 = px.scatter(
            filtered_df, 
            x='IMDB_Rating', 
            y='Gross', 
            hover_data=['Series_Title', 'Director'], # Añadí Director al hover para cuando ves 'Todos'
            color='IMDB_Rating'
        )
        fig2.update_layout(autosize=True, xaxis_title="IMDb Rating", yaxis_title="Recaudación USD")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Gemini Chat ---
    st.divider()
    st.subheader("🤖 Cine-AI")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pregunta sobre los datos filtrados..."):
        if not api_key:
            st.error("⚠️ Falta la API Key")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Preparar contexto optimizado (Smart Context)
        cols = ['Series_Title', 'Released_Year', 'IMDB_Rating', 'Gross', 'Director', 'Genre']
        cols = [c for c in cols if c in filtered_df.columns]
        
        ctx_df = filtered_df[cols].copy()
        
        # Si hay muchos datos (ej: Análisis Global), filtramos para no saturar
        warning_msg = ""
        if len(ctx_df) > 60:
            # Enviamos Top 30 taquilleras + Top 30 mejor valoradas para dar contexto variado
            top_gross = ctx_df.sort_values('Gross', ascending=False).head(30)
            top_rated = ctx_df.sort_values('IMDB_Rating', ascending=False).head(30)
            ctx_df = pd.concat([top_gross, top_rated]).drop_duplicates()
            warning_msg = "(Nota para la IA: El usuario está viendo muchos datos. Se envía una muestra representativa de las Top películas por recaudación y rating)."

        csv_txt = ctx_df.to_csv(index=False)
        
        context_desc = f"del director {selected_director}" if selected_director != "Todos" else "de una selección global de películas"
        
        sys_prompt = f"""Eres un experto en cine. El usuario está analizando datos {context_desc}.
        Datos disponibles (CSV):
        {csv_txt}
        {warning_msg}
        Responde a la pregunta basándote en estos datos. Si preguntan algo general fuera de los datos, usa tu conocimiento general pero avisa."""

        with st.chat_message("assistant"):
            status = st.status("Analizando...", expanded=True)
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=[sys_prompt, prompt]
                )
                status.update(label="Listo", state="complete", expanded=False)
                
                def streamer():
                    for chunk in response:
                        if chunk.text: yield chunk.text
                        
                full_res = st.write_stream(streamer())
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Error AI: {e}")
