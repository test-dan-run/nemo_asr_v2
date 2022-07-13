import os
import json

from typing import Tuple, Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig, OmegaConf, open_dict
from nemo.utils import logging

from nemo.utils.exp_manager import exp_manager
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models import EncDecRNNTBPEModel

from utils.hydra import hydra_runner
from utils.asr_tokenizer import build_document_from_manifests, process_tokenizer

def prepare_tokenizer(cfg: DictConfig) -> DictConfig:

    # compute sentencepiece tokenizer if not imported from somewhere
    if not cfg.tokenizer.precomputed:
        tokenizer_data_root = os.path.abspath('./tokenizer')
        manifests = cfg.nemo.model.train_ds.manifest_filepath + ',' + cfg.nemo.model.validation_ds.manifest_filepath

        text_corpus_path = build_document_from_manifests(tokenizer_data_root, manifests)
        cfg.nemo.model.tokenizer.dir = process_tokenizer(
            text_path = text_corpus_path,
            dst_folder = tokenizer_data_root,
            vocab_size = cfg.tokenizer.vocab_size,
            tokenizer_type = cfg.tokenizer.type,
            spe_type = cfg.tokenizer.spe_type,
            lower_case = True,
            spe_character_coverage= 1.0,
            spe_sample_size = -1,
            spe_train_extremely_large_corpus = False,
            spe_max_sentencepiece_length = -1,
            spe_bos = False,
            spe_eos = False,
            spe_pad = False,
        )

    return cfg

def prepare_model(cfg: DictConfig) -> Tuple[Any, pl.Trainer]:

    model_path = cfg.pretrained_model.path
    assert model_path.endswith('.nemo') or model_path.endswith('.ckpt')
    with open_dict(cfg):
        if model_path.endswith('.nemo'):
            cfg.nemo.init_from_nemo_model = model_path
        else:
            cfg.nemo.init_from_ptl_ckpt = model_path
    
    callbacks = []
    if cfg.early_stopping.patience > 0:
         callbacks.append(
            EarlyStopping(monitor='val_wer', mode='min', patience=cfg.early_stopping.patience),
        )

    trainer = pl.Trainer(**cfg.nemo.trainer, callbacks=callbacks)
    exp_manager(trainer, cfg.nemo.get("exp_manager", None))

    if 'transducer' in cfg.nemo.name.lower():
        asr_model = EncDecRNNTBPEModel(cfg=cfg.nemo.model, trainer=trainer)
    elif 'ctc' in cfg.nemo.name.lower():
        asr_model = EncDecCTCModelBPE(cfg=cfg.nemo.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg.nemo)

    return asr_model, trainer

# manifest_path should be located in the same directory as the audio files
# update manifest to update relative audio_filepaths to absolute paths
# returns new manifest path
def update_manifest_from_json(manifest_path: str) -> str:

    main_dir = os.path.dirname(manifest_path)
    new_manifest_path = os.path.join(
        main_dir, 
        'updated_' + os.path.basename(manifest_path)
        )
    with open(manifest_path, 'r') as fr, open(new_manifest_path, 'w') as fw:
        lines = fr.readlines()
        for line in lines:
            row = json.loads(line)
            row['audio_filepath'] = os.path.join(main_dir, row['audio_filepath'])
            fw.write(
                json.dumps(row) + '\n'
            )

    return new_manifest_path

# def test(cfg: DictConfig, asr_model: Any = None, trainer: pl.Trainer = None) -> None:

#     if asr_model is None and trainer is None:
#         cfg = prepare_dataset(cfg, test_only=True)
#         cfg = prepare_tokenizer(cfg)
#         asr_model, trainer = prepare_model(cfg)

#     if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
#         if asr_model.prepare_test(trainer):
#             trainer.test(asr_model)

#     else:
#         print('No test dataset found.')
#         return

#     return asr_model, trainer

# def transcribe(cfg: DictConfig, asr_model: Any = None, trainer: pl.Trainer = None, output_path: str = 'output.json') -> None:

#     if asr_model is None and trainer is None:
#         cfg = prepare_dataset(cfg, test_only=True)
#         cfg = prepare_tokenizer(cfg)
#         asr_model, trainer = prepare_model(cfg)

#     asr_model.set_trainer(trainer)
#     asr_model = asr_model.eval()
    
#     with open(cfg.model.test_ds.manifest_filepath, 'r', encoding='utf-8') as fr:
#         lines = fr.readlines()
    
#     items = [json.loads(item.strip('\r\n')) for item in lines]
#     audio_filepaths = [item['audio_filepath'] for item in items]
#     hypotheses = asr_model.transcribe(
#         audio_filepaths, 
#         batch_size = cfg.model.test_ds.batch_size,
#         num_workers = cfg.model.test_ds.num_workers)
#     for item, hypo in zip(items, hypotheses):
#         item['hypothesis'] = hypo

#     with open(output_path, 'w', encoding='utf-8') as fw:
#         for item in items:
#             fw.write(json.dumps(item) + '\n')

#     return output_path

@hydra_runner(config_path='./configs', config_name='main')
def main(cfg):
    asr_model, trainer = None, None

    if cfg.task_type == 'training':
        # direct manifest paths
        cfg.nemo.model.train_ds.manifest_filepath = update_manifest_from_json(cfg.dataset.train_ds_manifest_path)
        cfg.nemo.model.validation_ds.manifest_filepath = update_manifest_from_json(cfg.dataset.dev_ds_manifest_path)
        cfg.nemo.model.test_ds.manifest_filepath = update_manifest_from_json(cfg.dataset.test_ds_manifest_path)

        # set up tokenizer
        cfg = prepare_tokenizer(cfg)

        # prep model initialisation
        asr_model, trainer = prepare_model(cfg)

        trainer.fit(asr_model)
    # test(cfg, asr_model, trainer)

if __name__ == '__main__':
    main()