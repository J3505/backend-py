from flask import Flask, request, jsonify
from flask_cors import CORS  # Importamos CORS
import joblib  # Usamos joblib para cargar el modelo

app = Flask(__name__)

# Habilitar CORS para todas las rutas de la API
CORS(app)  # Esto permitirá solicitudes de cualquier origen (dominio)

# Cargar el modelo previamente entrenado
model = joblib.load('app/models/decision_tree_model.pkl')  # Ajusta la ruta según sea necesario

@app.route('/api/recommend', methods=['POST'])
def recommend_hotel():
    try:
        # Recibir los datos de la solicitud (en formato JSON)
        data = request.get_json()

        # Asegúrate de que todos los campos estén presentes en la solicitud
        required_fields = ['ubicacion', 'precio', 'servicios', 'calificacion', 'fechas', 'distancia', 'piso']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Falta el campo {field}'}), 400
        
        # Extraer los valores del JSON
        ubicacion = data['ubicacion']
        precio = data['precio']
        servicios = data['servicios']
        calificacion = data['calificacion']
        fechas = data['fechas']
        distancia = data['distancia']
        piso = data['piso']

        # Realizar la predicción con el modelo
        prediction = model.predict([[ubicacion, precio, servicios, calificacion, fechas, distancia, piso]])

        # Devolver la recomendación (en este caso, la calificación predicha)
        return jsonify({'recomendacion': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
