import mlflow

mlflow.set_tracking_uri(uri='http://192.168.164.14:5000')
print(mlflow.models.list_evaluators())
