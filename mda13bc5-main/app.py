# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

def estilo_fig(fig):
    fig.update_layout(
        template="plotly_white",
        title={
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20}
        },
        font=dict(size=12),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_traces(
        marker=dict(color="#636EFA"),
        hovertemplate="%{y:.2f} minutos"
    )

    return fig

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un analista de datos experto en hábitos de escucha de Spotify.

Trabajas con un dataframe de pandas llamado df. Cada fila representa una reproducción de una canción.

El dataset cubre desde {fecha_min} hasta {fecha_max}.

Columnas disponibles en df:
- ts: fecha y hora de la reproducción
- ms_played: milisegundos escuchados
- minutes_played: minutos escuchados
- master_metadata_track_name: nombre de la canción
- master_metadata_album_artist_name: artista principal
- master_metadata_album_album_name: álbum
- spotify_track_uri: identificador único de la canción
- reason_start: motivo de inicio
- reason_end: motivo de fin
- shuffle: si estaba activado el modo aleatorio
- skipped: si la canción fue saltada
- platform: plataforma o dispositivo usado
- year: año
- month: número de mes
- month_name: nombre del mes en español
- day_of_week: número de día de la semana (0=Lunes, 6=Domingo)
- day_name: nombre del día de la semana en español
- hour: hora del día (0 a 23)
- is_weekend: indica si es fin de semana
- week_part: "Entre semana" o "Fin de semana"
- semester: "S1" o "S2"
- season: "Invierno", "Primavera", "Verano" u "Otoño"

Valores observados:
- platform: {plataformas}
- reason_start: {reason_start_values}
- reason_end: {reason_end_values}

Tu tarea es responder preguntas del usuario sobre sus hábitos de escucha generando código Python que se ejecutará localmente sobre el dataframe df.

Tipos de preguntas esperadas:
1. Rankings y favoritos: artista más escuchado, top canciones, top artistas
2. Evolución temporal: escucha por mes, por semana, por día
3. Patrones de uso: hora del día, día de la semana, entre semana vs fin de semana
4. Comportamiento de escucha: shuffle, canciones saltadas
5. Comparaciones entre periodos: verano vs invierno, primer semestre vs segundo semestre

REGLAS CRÍTICAS:
- No inventes columnas que no existen
- Usa únicamente el dataframe df
- Prioriza las columnas derivadas ya preparadas: minutes_played, month_name, day_name, hour, week_part, semester y season
- No vuelvas a crear columnas si ya existen
- No uses información externa
- No leas archivos
- Devuelve SIEMPRE un JSON válido
- No incluyas markdown ni texto fuera del JSON
- El código debe ser ejecutable tal cual
- Si la pregunta pide análisis visual, el código debe crear SIEMPRE una figura de Plotly y guardarla en una variable llamada fig
- No devuelvas solo una serie, una fila o un dataframe como resultado final
- Si la pregunta no se puede responder con este dataset, devuelve tipo "texto" y explica la limitación
- Para rankings, evolución temporal, patrones y comparaciones, usa "grafico"

REGLAS DE CÓDIGO:
- Importa las librerías necesarias dentro del código
- Usa pandas como pd
- Usa plotly.express como px
- La figura final debe quedar guardada en fig
- No uses matplotlib, seaborn ni streamlit
- No uses print como salida principal
- Para tiempo escuchado, usa preferentemente minutes_played
- Para preguntas de “más escuchado”, prioriza minutes_played salvo que la pregunta pida explícitamente “más veces”
- Respeta el idioma español en títulos, ejes e interpretación
- Cuando uses month_name, day_name o season, conserva su orden natural
- Si filtras valores categóricos, usa exactamente estos nombres:
  - week_part: "Entre semana", "Fin de semana"
  - semester: "S1", "S2"
  - season: "Invierno", "Primavera", "Verano", "Otoño"

GUÍAS DE VISUALIZACIÓN:
- Usa barras para top artistas, top canciones, estaciones, meses, días, week_part y semester
- Usa líneas solo cuando tenga sentido para evolución temporal continua
- Para “¿cuál es mi artista más escuchado?” o “¿qué mes escucho más?”, puedes mostrar top 1 o una comparación visual breve, pero la interpretación debe responder la pregunta directamente

FORMATO DE RESPUESTA OBLIGATORIO:
Debes devolver exactamente un objeto JSON con estas claves:
- tipo: "grafico" o "texto"
- codigo: string con código Python ejecutable
- interpretacion: explicación breve, clara, directa y profesional en español

La interpretación DEBE:
- Responder directamente la pregunta del usuario
- Incluir explícitamente el resultado concreto (ej: "Julio", "Domingo", "Bad Bunny")
- Empezar por ese resultado (NO por explicación general)
- Añadir una breve interpretación del comportamiento del usuario
- Sonar como una conclusión analítica, no como descripción visual
- Tener máximo 2 frases
- No usar frases genéricas como "el que tiene mayor cantidad" sin decir cuál es
- Siempre que sea posible, incluye el nombre específico del resultado (mes, día, artista, etc.)
- La interpretación debe basarse en resultados calculados en el código, no en suposiciones generales
- No empezar con expresiones como “Este gráfico muestra…”, “La barra más alta…” o “Se observa que…”


Ejemplos de estilo de interpretación correcto:
- "El mes en el que más escuchas música es Julio, lo que indica un pico claro de consumo durante verano."
- "El día con mayor consumo es Domingo, lo que sugiere que escuchas más música en momentos de descanso."
- "Tu artista más escuchado es Bad Bunny, con una diferencia clara frente al resto."
- "Tu artista más escuchado es Taylor Swift, con una ventaja clara frente al resto de artistas."
- "Escuchas más música entre semana, lo que sugiere que tu consumo está más ligado a tu rutina diaria."

Ejemplo válido:
{{
  "tipo": "grafico",
  "codigo": "import pandas as pd\\nimport plotly.express as px\\ndf2 = df.groupby('master_metadata_album_artist_name', as_index=False)['minutes_played'].sum()\\ndf2 = df2.sort_values('minutes_played', ascending=False).head(10)\\nfig = px.bar(df2, x='master_metadata_album_artist_name', y='minutes_played', title='Top 10 artistas más escuchados', labels={{'master_metadata_album_artist_name': 'Artista', 'minutes_played': 'Minutos escuchados'}})",
  "interpretacion": "Tus artistas más escuchados están liderados por el que acumula más minutos reproducidos, lo que refleja una preferencia clara por ese grupo de artistas."
}}

Si la pregunta no puede responderse, devuelve algo como:
{{
  "tipo": "texto",
  "codigo": "",
  "interpretacion": "No puedo responder esa pregunta con las columnas disponibles en este dataset."
}}
"""





# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ================================
    # LIMPIEZA BÁSICA
    # ================================
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.dropna(subset=["master_metadata_track_name", "master_metadata_album_artist_name"])

    # ================================
    # MÉTRICAS
    # ================================
    df["minutes_played"] = df["ms_played"] / 60000

    # ================================
    # VARIABLES DE TIEMPO
    # ================================
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour

    month_map = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    day_map = {
        0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
        4: "Viernes", 5: "Sábado", 6: "Domingo"
    }

    df["month_name"] = df["month"].map(month_map)
    df["day_name"] = df["day_of_week"].map(day_map)

    # ================================
    # FIN DE SEMANA
    # ================================
    df["is_weekend"] = df["day_of_week"] >= 5
    df["week_part"] = df["is_weekend"].map({True: "Fin de semana", False: "Entre semana"})

    # ================================
    # SEMESTRE
    # ================================
    df["semester"] = df["month"].apply(lambda x: "S1" if x <= 6 else "S2")

    # ================================
    # ESTACIONES
    # ================================
    def get_season(month):
        if month in [12, 1, 2]:
            return "Invierno"
        elif month in [3, 4, 5]:
            return "Primavera"
        elif month in [6, 7, 8]:
            return "Verano"
        else:
            return "Otoño"

    df["season"] = df["month"].apply(get_season)

    # ================================
    # ORDEN CORRECTO DE CATEGORÍAS
    # ================================
    month_order = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]

    day_order = [
        "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"
    ]

    season_order = ["Invierno", "Primavera", "Verano", "Otoño"]

    df["month_name"] = pd.Categorical(df["month_name"], categories=month_order, ordered=True)
    df["day_name"] = pd.Categorical(df["day_name"], categories=day_order, ordered=True)
    df["season"] = pd.Categorical(df["season"], categories=season_order, ordered=True)

    # ================================
    # FIN DE SEMANA
    # ================================
    df["is_weekend"] = df["day_of_week"] >= 5
    df["week_part"] = df["is_weekend"].map({True: "Fin de semana", False: "Entre semana"})

    # ================================
    # SEMESTRE
    # ================================
    df["semester"] = df["month"].apply(lambda x: "S1" if x <= 6 else "S2")

    # ================================
    # ESTACIONES
    # ================================
    def get_season(month):
        if month in [12, 1, 2]:
            return "Invierno"
        elif month in [3, 4, 5]:
            return "Primavera"
        elif month in [6, 7, 8]:
            return "Verano"
        else:
            return "Otoño"

    df["season"] = df["month"].apply(get_season)

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["MDA13BC5"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        fig = estilo_fig(fig)  # 👈 ESTA ES LA LÍNEA NUEVA
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"**📊 Insight:** {parsed['interpretacion']}")
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    [Tu respuesta aquí]
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    [Tu respuesta aquí]
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    [Tu respuesta aquí]