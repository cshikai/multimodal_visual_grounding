
import os
from typing import Dict

from clearml import Task, Dataset

from config.config import cfg

Task.force_requirements_env_freeze(
    force=True, requirements_file="requirements.txt")
Task.add_requirements("torch")


def download_datasets(cfg: Dict) -> None:
    TEMP_PATH = '/tmp'
    for dataset in cfg['data']['clearml_datasets']:
        dataset_id = cfg['data']['clearml_datasets'][dataset]['id']
        clearml_dataset = Dataset.get(dataset_id=dataset_id)
        local_root_path = clearml_dataset.get_mutable_local_copy(
            os.path.join(TEMP_PATH, dataset_id))
        dataset_path = cfg['data']['clearml_datasets'][dataset]['path']
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        for f in os.listdir(local_root_path):
            full_f = os.path.join(local_root_path, f)
            new_f = os.path.join(dataset_path, f)
            os.rename(full_f, new_f)


def download_models(cfg: Dict) -> None:

    for model in cfg['training']['input_models']:
        model_id = cfg['training']['input_models'][model]['id']
        model_task = Task.get_task(task_id=model_id)
        last_snapshot = model_task.models['output'][-1]
        local_weights_path = last_snapshot.get_local_copy(
            os.path.join('/models', model))
        cfg['training']['input_models'][model]['path'] = local_weights_path


if __name__ == '__main__':
    PROJECT_NAME = cfg['clearml']['project_name']
    TASK_NAME = cfg['clearml']['task_name']
    OUTPUT_URL = cfg['clearml']['output_url']

    # Task.add_requirements("git+https://github.com/huggingface/datasets.git")
    # Task.add_requirements("hydra-core")
    # Task.add_requirements("pytorch-lightning")

    task = Task.init(project_name=PROJECT_NAME,
                     task_name=TASK_NAME, output_uri=OUTPUT_URL)
    task.set_base_docker(
        docker_image=cfg['clearml']['base_image'],
        #     # docker_setup_bash_script=[
        #     #     'pip install pandas',
        #     # ]
    )
    task.connect(cfg)
    task.execute_remotely(
        queue_name=cfg['clearml']['queue'], exit_process=True)
    # print('done')

    # from experiment import Experiment
    # from torch.multiprocessing import set_start_method
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass

    # download_models(cfg)
    # download_datasets(cfg)
    print(cfg)
    # exp = Experiment(cfg, task)
    # exp.run_experiment()
    # exp.create_torchscript_model('class_model_v2.ckpt')
