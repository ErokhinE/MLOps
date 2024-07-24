from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow
import json
import pandas as pd


app = Flask(__name__)

# Load the model from the MLflow model registry
model_name = "gradient_boosting_regressor"
alias = "champion"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data['features']
        df = pd.DataFrame([features]).astype({ 'year': 'int64', 'make': 'int64', 'model': 'int64', 'trim': 'int64', 'body': 'int64', 'transmission': 'int64', 'state': 'int64', 'condition': 'float64', 'odometer': 'float64', 'color': 'int64', 'interior': 'int64', 'seller': 'int64', 'mmr': 'float64', 'saledate': 'int64' })

        prediction = model.predict(df)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/info', methods=['GET'])
def info():
    try:
        client = mlflow.tracking.MlflowClient()
        model_details = client.get_registered_model(model_name)
        model_info = {
            'name': model_details.name,
            'creation_timestamp': model_details.creation_timestamp,
            'last_updated_timestamp': model_details.last_updated_timestamp,
            'latest_versions': [{ 'version': v.version, 'current_stage': v.current_stage } for v in model_details.latest_versions]
        }
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
