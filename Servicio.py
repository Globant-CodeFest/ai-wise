from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import save_model
import joblib

app = Flask(__name__)

# Ruta para agregar un nuevo dato
@app.route('/api/prediccionLineal/<string:pais>/<int:mes>', methods=['GET'])
def prediccionLineal(pais,mes):

    # Cargar datos desde el archivo CSV
    data = pd.read_csv('1970-2021-totalaffect.csv', delimiter=',')

    #ps = 'CHL'
    ps = pais
    psFind = 'ISO_' + ps
    # Filtrar datos utilizando condiciones
    filtered_data = data[(data['ISO'] == ps)]

    print(filtered_data)

    # Separar las variables independientes (X) y la variable dependiente (y)
    X = data[['Month', 'ISO']]
    y = data['TotalAffected']

    # Codificar la variable categórica ISO
    X_encoded = pd.get_dummies(X, columns=['ISO'], drop_first=True)

    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_encoded, y)


    nuevo_dato = {
        'Month': [mes],
        psFind: [0]
    }
    df_nuevo = pd.DataFrame(nuevo_dato)
    prediccion = model.predict(df_nuevo)

    print(f"La predicción de TotalAffected para ISO {pais} en mes es:", prediccion[0])

    return jsonify({'valor': prediccion[0]})

# Ruta para agregar un nuevo dato
@app.route('/api/prediccionLogistica/<string:pais>/<int:mes>', methods=['GET'])
def prediccionLogistica(pais,mes):


    data = pd.read_csv('1970-2021-totalaffect.csv', delimiter=',', low_memory=False)

    ps = pais

    filtered_data = data[(data['ISO'] == ps)]

    X_train = data[['Month']].values
    y_train = data['TotalAffected'].values

    # Crear el modelo de regresión logística
    logistic_model = LogisticRegression(max_iter=1000)

    # Ajustar el modelo a los datos de entrenamiento
    print("inicio entrenemiento")
    logistic_model.fit(X_train, y_train)

    mes_prediction = mes
    X_prediction = [[mes_prediction]]

    print("inicio prediccion")
    # Realizar predicciones en los datos de prueba
    y_pred = logistic_model.predict(X_prediction)

    resultado = ""
    if y_pred[0] == 1:
        print(f"En el mes {mes_prediction} puede haber un desastre.")
        resultado = f"En el mes {mes_prediction} puede haber un desastre."
    else:
        print(f"En el mes {mes_prediction} es menos probable que ocurra un desastre.")
        resultado = f"En el mes {mes_prediction} puede haber un desastre."
    return resultado





    return jsonify({'valor': y_pred[0]})

# Ejecutar el servidor
if __name__ == '__main__':
    app.run(debug=True)