import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_and_save_model(data_path, output_dir):
    # Descargar recursos necesarios de nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    print("NLTK resources downloaded.")

    # Función de preprocesamiento de texto
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words]
        return ' '.join(tokens)

    # Cargar los datos
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Verificar si los datos se cargaron correctamente
    print("Data loaded. Showing first few rows:")
    print(df.head())

    # Eliminar filas con valores nulos en las columnas relevantes
    df = df.dropna(subset=['content', 'company_name', 'title', 'keywords', 'employment_type'])
    print("Dropped rows with null values.")

    # Aplicar preprocesamiento
    df['cleaned_content'] = df['content'].apply(preprocess_text)
    print("Text preprocessing done. Showing first few cleaned rows:")
    print(df[['content', 'cleaned_content']].head())

    # Vectorizar el texto
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_content'])
    print("Text vectorization done. Shape of X:", X.shape)

    # Codificar la variable objetivo
    le = LabelEncoder()
    df['employment_type_encoded'] = le.fit_transform(df['employment_type'])
    print("Label encoding done. Unique labels:", le.classes_)

    y = df['employment_type_encoded']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and test sets.")

    # Ajustar k_neighbors en SMOTE
    minority_class_count = y_train.value_counts().min()
    smote = SMOTE(random_state=42, k_neighbors=max(1, minority_class_count - 1))

    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print("SMOTE applied. New training set size:", X_train_sm.shape)

    # Definir el modelo
    model = LogisticRegression(max_iter=1000)

    # Definir los hiperparámetros a buscar
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }

    # Configurar el Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

    # Ajustar el modelo con los datos de entrenamiento
    grid_search.fit(X_train_sm, y_train_sm)

    # Obtener los mejores parámetros
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Usar los mejores parámetros para entrenar el modelo final
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_sm, y_train_sm)
    print("Model training complete with best parameters.")

    # Hacer predicciones y evaluar el modelo
    y_pred = best_model.predict(X_test)
    print("Predictions complete. Classification report:")

    # Obtener todas las etiquetas únicas
    all_labels = le.classes_
    report = classification_report(y_test, y_pred, labels=range(len(all_labels)), target_names=all_labels, zero_division=1)
    print(report)

    # Guardar el modelo y los objetos de preprocesamiento
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))
    print("Model and preprocessing objects saved.")

    # Guardar el reporte de clasificación
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved at {report_path}")

# Ejemplo de llamada a la función
# train_and_save_model('/path/to/your/data.csv', '/path/to/save/models')
