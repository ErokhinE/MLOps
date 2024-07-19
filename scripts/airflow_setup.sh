# Add the folders which contain source codes where you want Airflow to import them for pipelines
export PYTHONPATH=$PWD/src
# You can do this also from the code via appending to sys.path

# REPLACE <project-folder-path> with your project folder path
echo "export PYTHONPATH=$PWD/src" >> ~/.bashrc

# Run/Load the file content
source ~/.bashrc

# Activate the virtual environment again
source venv/bin/activate

# Create folders and files for logging the output of components
mkdir -p $AIRFLOW_HOME/logs $AIRFLOW_HOME/dags
echo > $AIRFLOW_HOME/logs/scheduler.log
echo > $AIRFLOW_HOME/logs/triggerer.log
echo > $AIRFLOW_HOME/logs/webserver.log

# Add log files to .gitignore
echo *.log >> $AIRFLOW_HOME/logs/.gitignore