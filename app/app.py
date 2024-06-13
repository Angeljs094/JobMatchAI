import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Descargar recursos necesarios de nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Asegurarse de que los stopwords están correctamente cargados
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Error loading stopwords: {e}")
    nltk.download('stopwords', force=True)
    stop_words = set(stopwords.words('english'))

# Función de preprocesamiento de texto
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(tokens)

@st.cache_data
def load_model():
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, vectorizer, label_encoder

@st.cache_data
def load_data():
    df_jobs = pd.read_csv('/home/angel/airflow/computrabajo_filter/computrabajo_jobs.csv')
    df_jobs = df_jobs.dropna(subset=['content', 'company_name', 'title', 'keywords', 'employment_type', 'url'])
    df_jobs['cleaned_content'] = df_jobs['content'].apply(preprocess_text)
    return df_jobs

# Cargar el modelo y los objetos de preprocesamiento
model, vectorizer, label_encoder = load_model()
df_jobs = load_data()

# Vectorizar los contenidos preprocesados
@st.cache_data
def vectorize_jobs(df_jobs):
    X_jobs = vectorizer.transform(df_jobs['cleaned_content'])
    df_jobs['predicted_employment_type'] = model.predict(X_jobs)
    return df_jobs, X_jobs

df_jobs, X_jobs = vectorize_jobs(df_jobs)

# Función para la búsqueda de empleo
def busqueda_empleo():
    st.header("Búsqueda de Empleo")

    # Registro del usuario
    st.subheader("Registro del Usuario")
    nombre = st.text_input("Nombre")
    apellido = st.text_input("Apellido")
    edad = st.number_input("Edad", min_value=18, max_value=100, step=1)
    experiencia = st.number_input("Años de Experiencia", min_value=0, max_value=50, step=1)

    # Selección de tecnologías y roles
    st.subheader("Tecnologías y Roles")
    lenguajes = st.multiselect("Lenguajes de Programación", ["Python", "Java", "C#", "C++", "JavaScript", "Ruby", "PHP", "Otro"])
    bases_datos = st.multiselect("Bases de Datos", ["MySQL", "SQL Server", "Oracle", "PostgreSQL", "SQLite", "MongoDB", "Otro"])
    frameworks = st.multiselect("Frameworks", ["Django", "Flask", "Spring", "ASP.NET", "React", "Angular", "Vue.js", "Otro"])
    clouds = st.multiselect("Plataformas Cloud", ["AWS", "Azure", "Google Cloud", "IBM Cloud", "Oracle Cloud", "Otro"])
    visualizadores = st.multiselect("Herramientas de Visualización", ["Power BI", "Tableau", "QlikView", "Looker", "Otro"])
    roles = st.multiselect("Roles", ["Programador Web", "Backend", "Móviles", "Analista de Datos", "Ingeniero de Datos"])

    # Convertir selección a texto
    tecnologias = " ".join(lenguajes + bases_datos + frameworks + clouds + visualizadores + roles)

    # Búsqueda de empleo
    st.subheader("Buscar Empleo")
    if st.button("Buscar"):
        # Preprocesar la descripción de tecnologías
        descripcion = preprocess_text(tecnologias)

        # Vectorizar el texto preprocesado
        X_new = vectorizer.transform([descripcion])

        # Hacer predicciones
        predicted_label = model.predict(X_new)
        predicted_employment_type = label_encoder.inverse_transform(predicted_label)[0]

        # Filtrar ofertas de trabajo por el tipo de empleo predicho y experiencia
        def filtrar_experiencia(row):
            exp_palabras = ['junior', 'semi-senior', 'senior']
            content_lower = row['content'].lower()
            if experiencia < 2:
                if any(palabra in content_lower for palabra in exp_palabras[1:]):
                    return False
            elif 2 <= experiencia <= 5:
                if 'senior' in content_lower:
                    return False
            return True

        filtered_jobs = df_jobs[df_jobs['predicted_employment_type'] == predicted_label[0]]
        filtered_jobs = filtered_jobs[filtered_jobs.apply(filtrar_experiencia, axis=1)]

        # Calcular la similitud con el perfil del usuario
        similitudes = cosine_similarity(X_new, vectorizer.transform(filtered_jobs['cleaned_content'])).flatten()
        filtered_jobs['similaridad'] = similitudes

        # Obtener los 10 trabajos más similares
        top_jobs = filtered_jobs.sort_values(by='similaridad', ascending=False).head(10)

        # Mostrar resultados
        st.write(f"Ofertas de trabajo recomendadas para el tipo de empleo: {predicted_employment_type}")
        for index, row in top_jobs.iterrows():
            st.write(f"**{row['title']}** en {row['company_name']}")
            st.write(f"{row['content'][:200]}...")  # Mostrar solo los primeros 200 caracteres del contenido
            st.write(f"[Ver más]({row['url']})")
            st.write("---")

# Función para el análisis de tendencias
def analisis_tendencias():
    st.header("Análisis de Tendencias de Empleo")

    # Selección de categoría de tecnologías
    categoria = st.selectbox("Selecciona una Categoría", ["Lenguajes de Programación", "Bases de Datos", "Frameworks", "Plataformas Cloud", "Herramientas de Visualización"])

    if categoria == "Lenguajes de Programación":
        tecnologias = ["Python", "Java", "C#", "C++", "JavaScript", "Ruby", "PHP", "Otro"]
    elif categoria == "Bases de Datos":
        tecnologias = ["MySQL", "SQL Server", "Oracle", "PostgreSQL", "SQLite", "MongoDB", "Otro"]
    elif categoria == "Frameworks":
        tecnologias = ["Django", "Flask", "Spring", "ASP.NET", "React", "Angular", "Vue.js", "Otro"]
    elif categoria == "Plataformas Cloud":
        tecnologias = ["AWS", "Azure", "Google Cloud", "IBM Cloud", "Oracle Cloud", "Otro"]
    elif categoria == "Herramientas de Visualización":
        tecnologias = ["Power BI", "Tableau", "QlikView", "Looker", "Otro"]

    # Selección de tecnologías para comparar
    st.subheader("Comparar Frecuencia de Tecnologías")
    tecnologias_comparar = st.multiselect(f"Selecciona Tecnologías ({categoria})", tecnologias)
    
    if tecnologias_comparar:
        keyword_freq = {tech: 0 for tech in tecnologias_comparar}
        for tech in tecnologias_comparar:
            keyword_freq[tech] = df_jobs['keywords'].str.contains(tech, case=False).sum()
        
        st.subheader("Frecuencia de Tecnologías")
        fig, ax = plt.subplots()
        ax.bar(keyword_freq.keys(), keyword_freq.values(), color='blue')
        ax.set_xlabel("Tecnologías")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Frecuencia de Tecnologías Seleccionadas en {categoria}")
        st.pyplot(fig)
    
    # Selección de tecnologías para ver la tendencia en el tiempo
    st.subheader("Tendencia de Tecnologías en el Tiempo")
    tecnologias_tendencia = st.multiselect(f"Selecciona Tecnologías para la Tendencia en {categoria}", tecnologias)

    if tecnologias_tendencia:
        tendencia_data = df_jobs.copy()
        tendencia_data['date'] = pd.to_datetime(tendencia_data['year'].astype(str) + '-' + tendencia_data['month'].astype(str), errors='coerce').dt.to_period('M')
        tendencia_data = tendencia_data.dropna(subset=['date'])
        tendencia_data = tendencia_data.set_index('date')

        fig, ax = plt.subplots()
        for tech in tecnologias_tendencia:
            tech_tendencia = tendencia_data[tendencia_data['keywords'].str.contains(tech, case=False)].resample('M').size()
            ax.plot(tech_tendencia.index.to_timestamp(), tech_tendencia, label=tech)
        
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Número de Publicaciones")
        ax.set_title(f"Tendencia de Tecnologías en el Tiempo en {categoria}")
        ax.legend()
        st.pyplot(fig)

# Configuración de la barra lateral
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Selecciona una opción", ["Búsqueda de Empleo", "Análisis de Tendencias"])

if opcion == "Búsqueda de Empleo":
    busqueda_empleo()
elif opcion == "Análisis de Tendencias":
    analisis_tendencias()
