import hydra


def init_hydra():
    hydra.initialize(config_path="../configs")
    cfg = hydra.compose(config_name="main")
    return cfg