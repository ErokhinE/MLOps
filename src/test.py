import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(version_base=None, config_path='../configs', config_name='config')
def test(cfg: DictConfig):
    print(cfg.data.sample_size)


if __name__== '__main__':
    test()