from clearml import Task, Dataset, Model
from utils.hydra import hydra_runner
import os

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

    dataset_splits = ['train', 'dev', 'test']
    for split in dataset_splits:
        clearml_dataset = Dataset.get(dataset_id=cfg.dataset[f'{split}_clearml_id'])
        split_dir = clearml_dataset.get_local_copy()
        cfg.dataset[f'{split}_ds_manifest_path'] = os.path.join(split_dir, f'{split}_manifest.json')

    from local_main import prepare_model, prepare_tokenizer, update_manifest_from_json
    asr_model, trainer = None, None

    if cfg.task_type == 'training':

        # direct manifest paths
        cfg.nemo.model.train_ds.manifest_filepath = update_manifest_from_json(cfg.dataset.train_ds_manifest_path)
        cfg.nemo.model.validation_ds.manifest_filepath = update_manifest_from_json(cfg.dataset.dev_ds_manifest_path)
        cfg.nemo.model.test_ds.manifest_filepath = update_manifest_from_json(cfg.dataset.test_ds_manifest_path)

        # set up tokenizer
        cfg = prepare_tokenizer(cfg)
        task.upload_artifact(name='tokenizer_model', artifact_object=os.path.join(cfg.nemo.model.tokenizer.dir, 'tokenizer.model'))
        task.upload_artifact(name='tokenizer_vocab', artifact_object=os.path.join(cfg.nemo.model.tokenizer.dir, 'tokenizer.vocab'))

        # prep model initialisation
        asr_model, trainer = prepare_model(cfg)

        trainer.fit(asr_model)

if __name__ == '__main__':
    main()