import os
import json

from typing import Tuple, Any, List, Dict, Union
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

    if cfg.get('pretrained_model', None):
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

def extract_char_counts_in_items(items: List[Dict[str, Any]]) -> Dict[str, int]:
    
    char2count = {}
    for item in items:
        text = item['text']
        for ch in text:
            if ch not in char2count:
                char2count[ch] = 0
            char2count[ch] += 1
            
    return char2count

def characters_in_text(text: str, characters:Union[List[str], str]) -> bool:
    return any([ch in characters for ch in text])

def clean_transcripts_in_manifest(
    artifact_path: str, output_path: str, 
    remove_punct: bool, remove_english: bool, 
    remove_insuff_chars: bool, insuff_thres: Union[int, float] = 0.01
    ) -> str:

    with open(artifact_path, mode='r', encoding='utf-8') as fr:
        lines = fr.readlines()
    items = [json.loads(line) for line in lines]

    if remove_punct:
        items = [remove_punctuations_in_item(item) for item in items]
    if remove_english:
        items = [item for item in items if not characters_in_text(item['text'], ENGLISH_CHARACTERS)]
    if remove_insuff_chars:
        char2count = extract_char_counts_in_items(items)
        min_count = insuff_thres if type(insuff_thres) is int else insuff_thres*len(items)
        char_to_remove = [k for k in char2count.keys() if char2count[k] < min_count]
        items = [item for item in items if not characters_in_text(item['text'], char_to_remove)]

    with open(output_path, mode='w', encoding='utf-8') as fw:
        for item in items:
            fw.write(json.dumps(item)+'\n')

    return output_path

def update_manifests_from_json(manifest_paths: List[str], remove_unused_chars: bool = True, threshold: int = 50):
    
    path2items, char2count = {}, {}

    for path in manifest_paths:
        with open(path, 'r') as fr:
            lines = fr.readlines()
        items = [json.loads(line) for line in lines]
        path2items[path] = items

        # get character counts
        for item in items:
            for ch in item['text']:
                if ch not in char2count:
                    char2count[ch] = 0
                char2count[ch] += 1

    print('=================')
    print(f'Total items from {len(manifest_paths)} manifest paths: {sum([len(x) for x in path2items.values()])}')

    if remove_unused_chars:
        char2remove = [k for k in char2count.keys() if char2count[k] < threshold]
        for path in manifest_paths:
            path2items[path] = [item for item in path2items[path] if not characters_in_text(item['text'], char2remove)]

    print(f'Total items after removal: {sum([len(x) for x in path2items.values()])}')
    print(f'Logging counts of all characters:')
    for k,v in char2count.items():
        print(f'{k}: {v}')
    print('=================')

    outputs = []
    for path, items in path2items.items():
        outputs.append(update_manifest_paths_from_items(path, items))

    return outputs

def update_manifest_paths_from_items(manifest_path: str, items: List[Dict[str, Any]]) -> str:

    main_dir = os.path.dirname(manifest_path)
    new_manifest_path = os.path.join(
        main_dir, 
        'updated_' + os.path.basename(manifest_path)
        )
    
    with open(new_manifest_path, 'w') as fw:
        for item in items:
            item['audio_filepath'] = os.path.join(main_dir, item['audio_filepath'])
            fw.write(
                json.dumps(item) + '\n'
            )

    return new_manifest_path

# manifest_path should be located in the same directory as the audio files
# update manifest to update relative audio_filepaths to absolute paths
# returns new manifest path
def update_manifest_paths_from_json(manifest_path: str) -> str:

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

@hydra_runner(config_path='./configs', config_name='main')
def main(cfg):
    asr_model, trainer = None, None

    if cfg.task_type == 'training':
        # direct manifest paths
        cfg.nemo.model.train_ds.manifest_filepath = update_manifest_paths_from_json(cfg.dataset.train_ds_manifest_path)
        cfg.nemo.model.validation_ds.manifest_filepath = update_manifest_paths_from_json(cfg.dataset.dev_ds_manifest_path)
        cfg.nemo.model.test_ds.manifest_filepath = update_manifest_paths_from_json(cfg.dataset.test_ds_manifest_path)

        # set up tokenizer
        cfg = prepare_tokenizer(cfg)

        # prep model initialisation
        asr_model, trainer = prepare_model(cfg)

        trainer.fit(asr_model)
    # test(cfg, asr_model, trainer)

    if cfg.task_type == 'testing':
        cfg.nemo.model.train_ds.manifest_filepath = None
        cfg.nemo.model.validation_ds.manifest_filepath = None
        cfg.nemo.model.test_ds.manifest_filepath = update_manifest_paths_from_json(cfg.dataset.test_ds_manifest_path)
        cfg.nemo.model.tokenizer.dir = '/nemo_asr_v2/pretrained_model/'
        
        asr_model, trainer = prepare_model(cfg)

        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)

if __name__ == '__main__':
    main()