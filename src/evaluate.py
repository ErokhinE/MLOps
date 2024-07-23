from model import retrieve_model_with_alias, load_features
from sklearn.metrics import root_mean_squared_error
import hydra
import numpy as np
import mlflow

mlflow.set_tracking_uri(uri='http://192.168.164.14:5000')
def run(config):
    data_version = config.data_version
    X, y = load_features(dataset_name="final_features_target", dataset_version=data_version)
    model = retrieve_model_with_alias(config.model_name, config.model_alias)
    predictions = model.predict(X)
    print('Evaluation of', config.model_name, config.model_alias)
    print("rmse: ", root_mean_squared_error(y, predictions))


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg=None):
    run(cfg)


if __name__ == "__main__":
    main()
