import giskard
from model import load_features, retrieve_model_with_alias
from sklearn.metrics import mean_squared_error
import os
import numpy as np
import pandas as pd
import joblib

def test_champion_model(name='gradient_boosting_regressor_champion'):
    proj_path = os.environ['PROJECT_DIR']
    model = joblib.load(open(f'{proj_path}/models/{name}.joblib', 'rb')) 
    giskard_model = giskard.Model(
            model=model.predict,  # load_model is a custom function to load your model
            name=f"{name}_champion",
            model_type='regression',
            target_name='sellingprice',
        )
    test_data = load_features('', 2)
    X, y = test_data
    df = pd.DataFrame(X)
    df['sellingprice'] = y
    giskard_dataset = giskard.Dataset(
        df=df,
        name="test_datasetv2",
        target="sellingprice"
    )
    success_threshold = -1600.0
    suite_name = f"suite_{giskard_model.name}_{giskard_dataset.name}_2"
    test_suite = giskard.Suite(suite_name)
    def passed_the_neg_root_mean_squared_error(model, dataset):
            y_true, y_pred = dataset.df[dataset.target], model.predict(dataset).raw
            neg_root_mean_squred_error = -np.sqrt(mean_squared_error(y_true, y_pred))
            return giskard.TestResult(passed=neg_root_mean_squred_error>=success_threshold)
    test_suite.add_test(passed_the_neg_root_mean_squared_error, model=giskard_model, dataset=giskard_dataset, test_id='neg_root_mean_squared_error')
    result = test_suite.run()
    return result.passed

def main():
    if test_champion_model():
        print('Success')
    else:
        raise ValueError('Champion model performs bad')
    
if __name__ == '__main__':
     main()
    
