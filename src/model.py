import importlib
import random

import mlflow
import numpy as np
import pandas as pd
from zenml.client import Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import mlflow.pyfunc

mlflow.set_tracking_uri(uri='http://192.168.164.14:8080')

def load_features(dataset_name, dataset_version, sample_fraction=1):
    client = Client()
    artifacts = client.list_artifacts(name=dataset_name, version=dataset_version)
    sorted_artifacts = sorted(artifacts, key=lambda artifact: artifact.version, reverse=True)
    dataframe = sorted_artifacts[0].load()
    sampled_dataframe = dataframe.sample(
        frac=sample_fraction, random_state=88
    )
    feature_data = sampled_dataframe[sampled_dataframe.columns[:-1]]
    target_data = sampled_dataframe[sampled_dataframe.columns[-1]]
    return feature_data, target_data



def train(X_train, y_train, cfg):
    random.seed(123)
    np.random.seed(123)
    params = cfg.model.params
    estimator = RandomForestRegressor()
    param_grid = dict(params)
    scoring = list(cfg.model.metrics.values())
    evaluation_metric = cfg.model.evaluation_metric
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        refit=evaluation_metric,
        cv=cfg.model.folds,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)

    return gs


def log_metadata(config, grid_search, train_features, train_target, test_features, test_target):
    cross_val_results = (
        pd.DataFrame(grid_search.cv_results_)
        .filter(regex=r"std_|mean_|param_")
        .sort_index(axis=1)
    )
    optimal_metrics_values = [
        result[1][grid_search.best_index_] for result in grid_search.cv_results_.items()
    ]
    optimal_metrics_keys = [metric for metric in grid_search.cv_results_]
    optimal_metrics_dict = {
        key: value
        for key, value in zip(optimal_metrics_keys, optimal_metrics_values)
        if "mean" in key or "std" in key
    }

    hyperparameters = optimal_metrics_dict

    train_dataframe = pd.concat([train_features, train_target], axis=1)
    test_dataframe = pd.concat([test_features, test_target], axis=1)

    experiment_title = config.model.model_name + "_experiment_model"

    try:
        experiment_id = mlflow.create_experiment(name=experiment_title)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_title).experiment_id

    print("Experiment ID: ", experiment_id)

    cv_metric = config.model.cv_evaluation_metric
    run_title = "_".join(['model_run', config.model.model_name, config.model.evaluation_metric, str(hyperparameters[cv_metric]).replace(".", "_")])
    print("Run title: ", run_title)

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        pass

    with mlflow.start_run(run_name=run_title, experiment_id=experiment_id) as parent_run:
        train_dataset = mlflow.data.pandas_dataset.from_pandas(df=train_dataframe, targets='sellingprice')
        test_dataset = mlflow.data.pandas_dataset.from_pandas(df=test_dataframe, targets='sellingprice')
        mlflow.log_input(train_dataset, "training")
        mlflow.log_input(test_dataset, "testing")

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(optimal_metrics_dict)

        mlflow.set_tag('regressor', 'random_forest')
        model_signature = mlflow.models.infer_signature(train_features, grid_search.predict(train_features))

        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search.best_estimator_,
            artifact_path='random_forest_model',
            signature=model_signature,
            input_example=train_features,
            registered_model_name=config.model.model_name,
            pyfunc_predict_fn=config.model.pyfunc_predict_fn,
        )

        mlflow_client = mlflow.client.MlflowClient()
        mlflow_client.set_model_version_tag(
            name=config.model.model_name,
            version=model_info.version,
            key="source",
            value="best_grid_search_model",
        )

        test_predictions = grid_search.best_estimator_.predict(test_features)
        evaluation_data = pd.DataFrame(test_target)
        evaluation_data.columns = ["label"]
        evaluation_data["predictions"] = test_predictions

        evaluation_results = mlflow.evaluate(
            data=evaluation_data,
            model_type="regressor",
            targets="label",
            predictions="predictions",
            evaluators=["default"],
        )

        mlflow.log_metrics(evaluation_results.metrics)

        print(f"Best model metrics:\n{evaluation_results.metrics}")

        for idx, result in cross_val_results.iterrows():
            child_run_title = "_".join(["child", run_title, str(idx)])
            
            with mlflow.start_run(run_name=child_run_title, experiment_id=experiment_id, nested=True):
                child_params = result.filter(regex="param_").to_dict()
                child_means = result.filter(regex="mean_").to_dict()
                child_stds = result.filter(regex="std_").to_dict()
                child_params = {key.replace("param_", ""): value for key, value in child_params.items()}
                mlflow.log_params(child_params)
                mlflow.log_metrics(child_means)
                mlflow.log_metrics(child_stds)


def retrieve_model_with_alias(model_name, model_alias):
    model_uri = f"models:/{model_name}/{model_alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
