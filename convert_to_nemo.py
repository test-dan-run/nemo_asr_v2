from typing import Tuple, Any
import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict

from nemo.utils.exp_manager import exp_manager
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models import EncDecRNNTBPEModel

from utils.hydra import hydra_runner

def prepare_model(cfg: DictConfig) -> Tuple[Any, pl.Trainer]:

    if cfg.get('pretrained_model', None):
        model_path = cfg.pretrained_model.path
        assert model_path.endswith('.nemo') or model_path.endswith('.ckpt')
        with open_dict(cfg):
            if model_path.endswith('.nemo'):
                cfg.nemo.init_from_nemo_model = model_path
            else:
                cfg.nemo.init_from_ptl_ckpt = model_path

    trainer = pl.Trainer(**cfg.nemo.trainer)
    exp_manager(trainer, cfg.nemo.get("exp_manager", None))

    if 'transducer' in cfg.nemo.name.lower():
        asr_model = EncDecRNNTBPEModel(cfg=cfg.nemo.model, trainer=trainer)
    elif 'ctc' in cfg.nemo.name.lower():
        asr_model = EncDecCTCModelBPE(cfg=cfg.nemo.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg.nemo)

    return asr_model, trainer

@hydra_runner(config_path='./configs', config_name='convert_nemo')
def main(cfg):

    print(cfg.nemo.model.tokenizer.dir)

    asr_model, trainer = prepare_model(cfg)
    asr_model.save_to('/nemo_asr_v2/best.nemo')

if __name__ == '__main__':
    main()
