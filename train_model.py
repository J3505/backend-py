import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib  # Para guardar el modelo entrenado

# Cargar los datos
df = pd.read_csv('./data/hoteles.csv')

# Preprocesamiento de datos
label_encoder = LabelEncoder()

# Convertir la ubicación y servicios a números
df['ubicacion'] = label_encoder.fit_transform(df['ubicacion'])
df['servicios'] = df['servicios'].apply(lambda x: label_encoder.fit_transform(x.split(';')).max())
df['fechas'] = df['fechas'].apply(lambda x: 1 if x == 'Festiva' else 0)  # Festiva = 1, No festiva = 0

# Variables predictoras (X) y variable objetivo (y)
X = df[['ubicacion', 'precio', 'servicios', 'calificacion', 'fechas', 'distancia', 'piso']]
y = df['calificacion']  # Queremos predecir la calificación

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)

# Evaluar el modelo
score = clf.score(X_test, y_test)
print(f'Accuracy del modelo: {score * 100:.2f}%')

# Guardar el modelo entrenado
joblib.dump(clf, 'decision_tree_model.pkl')
