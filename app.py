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

# --- Carga y Limpieza de Datos (Blindada) ---
@st.cache_data
def load_data():
    # 1. Intentar cargar desde archivo local (Plan B manual)
    local_file = "imdb_top_1000.csv"
    if os.path.exists(local_file):
        try:
            df = pd.read_csv(local_file)
            st.toast("📂 Datos cargados desde archivo local.", icon="✅")
            return clean_dataframe(df)
        except Exception as e:
            st.warning(f"Archivo local encontrado pero corrupto: {e}")

    # 2. Intentar cargar desde URLs espejo (Repositorios estables)
    urls = [
        # Fuente 1: Proyecto de Data Science de 'JaviRute' (4 años de antigüedad, rama master)
        "https://raw.githubusercontent.com/JaviRute/top_1000_movies-data_science_project/master/imdb_top_1000.csv",
        # Fuente 2: Proyecto de 'Elliott-dev' (3 años de antigüedad, rama master)
        "https://raw.githubusercontent.com/Elliott-dev/Top-1000-IMDB-Rated-Movies-Analysis/master/imdb_top_1000.csv",
        # Fuente 3: Fuente original de Krishna (rama main)
        "https://raw.githubusercontent.com/krishna-koly/IMDB_TOP_1000/main/imdb_top_1000.csv"
    ]
    
    for url in urls:
        try:
            # Usamos on_bad_lines para saltar filas corruptas si las hubiera
            df = pd.read_csv(url, on_bad_lines='skip')
            
            # Verificación rápida de que es el dataset correcto
            if 'Director' in df.columns and 'Gross' in df.columns:
                return clean_dataframe(df)
            
        except Exception as e:
            # Continuar al siguiente espejo sin detenerse
            continue 
            
    # Si llegamos aquí, todo falló
    return pd.DataFrame()

def clean_dataframe(df):
    """Función auxiliar para limpiar el dataframe una vez cargado"""
    try:
        df.columns = df.columns.str.strip() # Limpiar espacios en nombres
        
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
        st.error(f"Error limpiando datos: {e}")
        return pd.DataFrame()

# --- Ejecución de Carga ---
with st.spinner("Conectando con repositorios de datos..."):
    df = load_data()

# --- VALIDACIÓN DE EMERGENCIA ---
if df.empty:
    st.error("❌ ERROR CRÍTICO: No se pudo descargar el dataset.")
    st.markdown("""
    **Solución Manual:**
    1. Descarga el archivo CSV [desde aquí](https://github.com/krishna-koly/IMDB_TOP_1000/blob/main/imdb_top_1000.csv).
    2. Guárdalo en la carpeta de tu proyecto con el nombre `imdb_top_1000.csv`.
    3. Recarga esta página.
    """)
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Inserta tu clave aquí...")
    st.divider()
    
    # Filtros seguros con dropna()
    directors = sorted(df['Director'].dropna().unique())
    selected_director = st.selectbox("Selecciona Director", directors)
    
    genres_list = df['Genre'].dropna().unique()
    genres = sorted(list(set([g.strip() for sublist in genres_list for g in sublist.split(',')]))) if not df.empty else []
    selected_genres = st.multiselect("Filtrar por Género", genres)

# --- Filtrado ---
filtered_df = df[df['Director'] == selected_director]

if selected_genres:
    pattern = '|'.join(selected_genres)
    filtered_df = filtered_df[filtered_df['Genre'].astype(str).str.contains(pattern, case=False, na=False)]

# --- Dashboard ---
st.title(f"🎬 Dashboard: {selected_director}")

if filtered_df.empty:
    st.warning("No hay películas con esos filtros.")
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
        df_chart = filtered_df.sort_values('Gross', ascending=False).head(10)
        fig = px.bar(df_chart, x='Series_Title', y='Gross', color='Gross', color_continuous_scale='Greens')
        fig.update_layout(autosize=True, xaxis_title="", yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True) # Usamos container width para responsive

    with c2:
        st.subheader("⭐ Rating vs Dinero")
        fig2 = px.scatter(filtered_df, x='IMDB_Rating', y='Gross', hover_data=['Series_Title'], color='IMDB_Rating')
        fig2.update_layout(autosize=True, xaxis_title="IMDb", yaxis_title="USD")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Gemini Chat ---
    st.divider()
    st.subheader("🤖 Cine-AI")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pregunta sobre este director..."):
        if not api_key:
            st.error("⚠️ Falta la API Key")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Preparar contexto optimizado
        cols = ['Series_Title', 'Released_Year', 'IMDB_Rating', 'Gross', 'Director']
        cols = [c for c in cols if c in filtered_df.columns]
        
        ctx_df = filtered_df[cols].copy()
        if len(ctx_df) > 50:
            ctx_df = ctx_df.sort_values('IMDB_Rating', ascending=False).head(50)
            
        csv_txt = ctx_df.to_csv(index=False)
        
        sys_prompt = f"Eres un experto en cine. Analiza estos datos del director {selected_director}:\n{csv_txt}\nResponde la pregunta del usuario."

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
