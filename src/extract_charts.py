import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the tracking URI (MLflow server URL)
mlflow.set_tracking_uri("http://192.168.164.14:5000")

# Experiment and Parent Run ID
results_folder = "results"

# Ensure the results folder exists
os.makedirs(results_folder, exist_ok=True)

# Get the client
client = mlflow.tracking.MlflowClient()
for experiment_id, parent_run_id, model_name in [["431725057885973349", "72830497de52405bb697bc6797d43653", "random_forest_regressor"], ["544441921844855430", "b9fa16659b114559bdfd5807a3399706", "gradient_boosting_regressor"]]:
    # Get child runs
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
    )
    print(len(child_runs))
    for run in child_runs:
        run_data = client.get_run(run.info.run_id).data
        for metric, value in run_data.metrics.items():
            plt.figure()
            plt.bar(run.info.run_id, value)
            plt.title(f"Run ID: {run.info.run_id}\nMetric: {metric}")
            plt.xlabel("Run ID")
            plt.ylabel(metric)
            plt.tight_layout()
            plot_filename = f"{run.info.run_id}_{metric}.png"
            plt.savefig(os.path.join(results_folder, plot_filename))
            plt.close()

print("Metrics and plots have been saved in the results folder.")
