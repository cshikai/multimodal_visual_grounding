
from clearml import Dataset

PROJECT_NAME = "datasets/multimodal"
DATASET_NAME = "mscoco"
DATASET_ROOT = '/data/MSCOCO_root/'
OUTPUT_URL = "s3://experiment-logging/multimodal"


clearml_dataset = Dataset.create(
    dataset_project=PROJECT_NAME, dataset_name=DATASET_NAME)

clearml_dataset.add_files(DATASET_ROOT)
clearml_dataset.upload(show_progress=True,
                       verbose=True, output_url=OUTPUT_URL)
clearml_dataset.finalize()

print('Done')
