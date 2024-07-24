import giskard
from model import load_features, retrieve_model_with_alias
from sklearn.metrics import mean_squared_error
import os
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd


def choose_best_model():
    """
    Choosing the best model
    """
    client = MlflowClient()
    challenger_models = []
    for model_type in ['random_forest_regressor', 'gradient_boosting_regressor']:
        for alias in ['challenger1', 'challenger2']:
            challenger_models.append((retrieve_model_with_alias(model_type, alias), model_type, alias))
    test_data = load_features('', 2)
    X, y = test_data
    df = pd.DataFrame(X)
    df['sellingprice'] = y
    giskard_dataset = giskard.Dataset(
        df=df,
        name="test_datasetv2",
        target="sellingprice"
    )
    giskard_models = []
    for model_info in challenger_models:
        model = giskard.Model(
            model=model_info[0].predict,  # load_model is a custom function to load your model
            name=f"{model_info[1]}_{model_info[2]}",
            model_type='regression',
            target_name='sellingprice',
            
        )
        giskard_models.append((model,model_info[2], model_info[1]))
        
    success_threshold = -1600.0

    results = []
    # Create test suite and add performance test
    for giskard_model_info in giskard_models:
        giskard_model, alias, name = giskard_model_info
        def passed_the_neg_root_mean_squared_error(model, dataset):
            y_true, y_pred = dataset.df[dataset.target], model.predict(dataset).raw
            neg_root_mean_squred_error = -np.sqrt(mean_squared_error(y_true, y_pred))
            return giskard.TestResult(passed=neg_root_mean_squred_error>=success_threshold)
        scan_res = giskard.scan(model=giskard_model, dataset=giskard_dataset)
        report_path = f"reports/report_{giskard_model.name}_test_dataset_v2.html"
        scan_res.to_html(report_path)
        suite_name = f"suite_{giskard_model.name}_{giskard_dataset.name}_2"
        test_suite = giskard.Suite(suite_name)
        test_suite.add_test(passed_the_neg_root_mean_squared_error, model=giskard_model, dataset=giskard_dataset, test_id='neg_root_mean_squared_error')
        result = test_suite.run()
        results.append({
            "model": giskard_model,
            "passed": result.passed,
            "issues": scan_res.issues,
            "alias": alias,
            "name": name
        })

    passing_models = [r for r in results if r["passed"]]

    # Sort by the number of issues (ascending) and other criteria if necessary
    passing_models_sorted = sorted(passing_models, key=lambda x: len(x['issues']))

    # Select the model with the least issues
    selected_model = passing_models_sorted[0] if passing_models_sorted else None
    def find_model_by_version(model_name, model_alias):
        version = client.get_model_version_by_alias(model_name, model_alias)
        return version.version

    if selected_model:
        client.transition_model_version_stage(
            name=selected_model['name'],
            version=find_model_by_version(selected_model['name'], selected_model['alias']),
            stage="Production"
        )
    else:
        print('No best model found')



def main():
    choose_best_model()

if __name__ == '__main__':
    main()

