from flask import Flask, request, jsonify
import joblib  # Para cargar el modelo entrenado

app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = joblib.load('app/models/decision_tree_model.pkl')

@app.route('/api/recommend', methods=['POST'])
def recommend_hotel():
    # Recibir los datos de la solicitud
    data = request.get_json()  # Recibe los datos en formato JSON
    
    # Extraer los valores de la solicitud
    ubicacion = data['ubicacion']
    precio = data['precio']
    servicios = data['servicios']
    calificacion = data['calificacion']
    fechas = data['fechas']
    distancia = data['distancia']
    piso = data['piso']

    # Realizar la predicción
    prediction = model.predict([[ubicacion, precio, servicios, calificacion, fechas, distancia, piso]])

    # Enviar la respuesta con la recomendación
    return jsonify({'recomendacion': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
