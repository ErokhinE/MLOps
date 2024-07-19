import zenml
from zenml.pipelines import Pipeline
from zenml.steps import step, Output
from zenml.artifact_stores import LocalArtifactStore

# Initialize ZenML client
client = zenml.Client()

# Define a step that generates artifacts
@step
def generate_data() -> Output(data=zenml.artifacts.Artifact):
    data = [1, 2, 3]
    return data

@step
def process_data(data: zenml.artifacts.Artifact) -> None:
    print(f"Processing data: {data}")

# Define the pipeline
@Pipeline
def my_pipeline(generate_data, process_data):
    data = generate_data()
    process_data(data)

# Set up the artifact store (if not already configured)
artifact_store = LocalArtifactStore()  # Replace with your configured artifact store
client.add_artifact_store(artifact_store)

# Run the pipeline
pipeline = my_pipeline(generate_data=generate_data(), process_data=process_data())
pipeline.run()
