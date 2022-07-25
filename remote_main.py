from clearml import Task, Dataset, Model, StorageManager
from utils.hydra import hydra_runner
import os
import shutil

@hydra_runner(config_path='./configs', config_name='remote_main')
def main(cfg):

    task = Task.init(
        project_name=cfg.clearml.project_name, task_name=cfg.clearml.task_name, 
        output_uri=cfg.clearml.output_uri, task_type=cfg.task_type
    )
    task.set_base_docker(cfg.clearml.base_docker_image)
    # duplicate task, reset state, and send to remote clearml agent, exit process after this
    task.execute_remotely(queue_name=cfg.clearml.queue_name, clone=False, exit_process=True)

    # download model and datasets
    if cfg.get('pretrained_model', None):
        clearml_model = Model(model_id=cfg.pretrained_model.model_clearml_id)
        cfg.pretrained_model.path = clearml_model.get_local_copy()

        # rename checkpoint file to nemo file (only for model_weights)
        if cfg.pretrained_model.get('rename_nemo', None):
            new_model_path = cfg.pretrained_model.path.replace('.ckpt', '.nemo')
            os.rename(cfg.pretrained_model.path, new_model_path)
            cfg.pretrained_model.path = new_model_path
    
    from local_main import prepare_model, prepare_tokenizer, update_manifests_from_json, update_manifest_paths_from_json
    asr_model, trainer = None, None

    if cfg.task_type == 'training':

        dataset_splits = ['train', 'dev']
        for split in dataset_splits:
            clearml_dataset = Dataset.get(dataset_id=cfg.dataset[f'{split}_clearml_id'])
            split_dir = clearml_dataset.get_local_copy()
            cfg.dataset[f'{split}_ds_manifest_path'] = os.path.join(split_dir, f'{split}_manifest.json')

        # direct manifest paths
        train_path, val_path = update_manifests_from_json([
            cfg.dataset.train_ds_manifest_path,
            cfg.dataset.dev_ds_manifest_path,
            ], remove_unused_chars=cfg.dataset.remove_unused_chars, threshold=cfg.dataset.threshold)

        cfg.nemo.model.train_ds.manifest_filepath = train_path
        cfg.nemo.model.validation_ds.manifest_filepath = val_path
        # cfg.nemo.model.test_ds.manifest_filepath = update_manifest_from_json(cfg.dataset.test_ds_manifest_path)

        # set up tokenizer
        cfg = prepare_tokenizer(cfg)
        task.upload_artifact(name='tokenizer_model', artifact_object=os.path.join(cfg.nemo.model.tokenizer.dir, 'tokenizer.model'))
        task.upload_artifact(name='tokenizer_vocab', artifact_object=os.path.join(cfg.nemo.model.tokenizer.dir, 'tokenizer.vocab'))

        # prep model initialisation
        asr_model, trainer = prepare_model(cfg)

        trainer.fit(asr_model)

    if cfg.task_type == 'testing':

        clearml_dataset = Dataset.get(dataset_id=cfg.dataset['test_clearml_id'])
        split_dir = clearml_dataset.get_local_copy()
        cfg.dataset[f'test_ds_manifest_path'] = os.path.join(split_dir, f'test_manifest.json')

        cfg.nemo.model.train_ds.manifest_filepath = None
        cfg.nemo.model.validation_ds.manifest_filepath = None
        cfg.nemo.model.test_ds.manifest_filepath = update_manifest_paths_from_json(cfg.dataset.test_ds_manifest_path)

        # you need the trained tokenizer if you are restoring from a checkpoint
        # if cfg.pretrained_model.path.endswith('.ckpt'):
        tokenizer_model_path = Task.get_task(task_id=cfg.tokenizer.task_id).artifacts['tokenizer_model'].get_local_copy()
        tokenizer_vocab_path = Task.get_task(task_id=cfg.tokenizer.task_id).artifacts['tokenizer_vocab'].get_local_copy()

        tokenizer_data_root = os.path.abspath('./tokenizer')
        os.makedirs(tokenizer_data_root, exist_ok=True)

        shutil.copy(tokenizer_model_path, os.path.join(tokenizer_data_root, 'tokenizer.model'))
        shutil.copy(tokenizer_vocab_path, os.path.join(tokenizer_data_root, 'vocab.txt'))

        cfg.nemo.model.tokenizer.dir = tokenizer_data_root
        
        asr_model, trainer = prepare_model(cfg)

    if asr_model.prepare_test(trainer):
        trainer.test(asr_model)
    
if __name__ == '__main__':
    main()