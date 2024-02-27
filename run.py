import hydra

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def start(cfg):
    
    
    