import pandas as pd
import hydra
from omegaconf import DictConfig
import os

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def sample_data(cfg: DictConfig):
    data_url = cfg.data.url
    sample_size = cfg.data.sample_size
    sample_file = cfg.data.sample_file

    # Read the data
    df = pd.read_csv(data_url)

    # Sample the data
    sample_df = df.sample(frac=sample_size, random_state=42)

    # Ensure the samples directory exists
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)

    # Save the sampled data
    sample_df.to_csv(sample_file, index=False)
    print(f"Sampled data saved to {sample_file}")

if __name__ == "__main__":
    sample_data()
