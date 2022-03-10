import os
from pathlib import Path
from clearml import Dataset, Task

DATA_PROJECT_NAME = "datasets/multimodal"
DATASET_NAME = "VisualGround_flicker"  # "VG_MSCOCO_FLICKR"

TASK_NAME = "visual_grounding_training_data_generation"

DATASET_ROOT = '/data/processed/'

OUTPUT_URL = "s3://experiment-logging/multimodal"


task = Task.init(project_name=DATA_PROJECT_NAME, task_name=TASK_NAME)

# task.set_base_docker(docker_image="dleongsh/audio_preproc:v1.0.0")
task.set_base_docker(
    docker_image="python:3.8",
    docker_setup_bash_script=['apt-get update -y',
                              'apt-get upgrade -y',
                              'pip install pandas clearml fastparquet boto3 dask',
                              ])

args = {
    'num_captions': 5,
    'npartitions': 1000,
    'batch_size': 32,
    'input_datasets': ['75c32c21d7b7490ba84e32d0a1593238','f5626d7f0cdb4b64807a6ff8f4e726d6','30c3d39e7bbf429e950225d8a062bcfb']
}

task.connect(args)
task.execute_remotely(queue_name='cpu-only')
from visual_grounding_dataset import VisualGroundingDataCreator
input_train_dataset_paths = []
input_valid_dataset_paths = []
input_image_folder_paths = []

TEMP_PATH = '/tmp'
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)
for dataset_id in args['input_datasets']:
    dataset = Dataset.get(dataset_id=dataset_id)
    local_root_path = dataset.get_mutable_local_copy(
        os.path.join(TEMP_PATH, dataset_id))

    input_train_dataset_paths.append(os.path.join(
        local_root_path, 'train_manifest.csv'))

    valid_path = os.path.join(
        local_root_path, 'valid_manifest.csv')

    image_folder = [os.path.join(local_root_path, name) for name in os.listdir(
        local_root_path) if os.path.isdir(os.path.join(local_root_path, name))][0]

    input_image_folder_paths.append(image_folder)
    if os.path.exists(valid_path):
        input_valid_dataset_paths.append(valid_path)



vg_data_creator = VisualGroundingDataCreator(
    num_captions=args['num_captions'], nparitions=args['npartitions'], batch_size=args['batch_size'])

processed_train_path = os.path.join(DATASET_ROOT, 'train')
processed_valid_path = os.path.join(DATASET_ROOT, 'valid')

vg_data_creator.create(input_train_dataset_paths, processed_train_path)
vg_data_creator.create(input_valid_dataset_paths, processed_valid_path)
for image_folder_path in input_image_folder_paths:
    os.rename(image_folder_path, os.path.join(
        DATASET_ROOT, Path(image_folder_path).stem))

clearml_dataset = Dataset.create(
    dataset_project=DATA_PROJECT_NAME, dataset_name=DATASET_NAME)

clearml_dataset.add_files(DATASET_ROOT)
clearml_dataset.upload(show_progress=True,
                       verbose=True, output_url=OUTPUT_URL)
clearml_dataset.finalize()


task.set_parameter(
    name="output_dataset_id",
    value=clearml_dataset.id,
    description="the dataset task id of the output dataset"
)
print('Done')
