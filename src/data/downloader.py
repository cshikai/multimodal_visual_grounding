import os
from pathlib import Path
from clearml import Dataset


class VisualGroundingDatasetDownloader():
    DATA_ROOT = '/data'
    TEMP_ROOT = '/temp'

    def __init__(self, cfg):
        self.cfg = cfg

        if not os.path.exists(self.DATA_ROOT):
            os.makedirs(self.DATA_ROOT)

        if not os.path.exists(self.TEMP_ROOT):
            os.makedirs(self.TEMP_ROOT)

    def download(self):

        # input_train_dataset_paths = []
        # input_valid_dataset_paths = []

        manifest_path = Dataset.get(dataset_id=self.cfg['manifest_dataset'])

        for data_mode in ['train', 'valid']:

            os.rename(os.path.join(manifest_path, data_mode), os.path.join(
                self.DATA_ROOT, data_mode))

        input_image_folder_paths = []

        for dataset_id in self.cfg['input_image_datasets']:
            dataset = Dataset.get(dataset_id=dataset_id)
            local_root_path = dataset.get_mutable_local_copy(
                os.path.join(self.TEMP_PATH, dataset_id))

            image_folder = [os.path.join(local_root_path, name) for name in os.listdir(
                local_root_path) if os.path.isdir(os.path.join(local_root_path, name))][0]

            input_image_folder_paths.append(image_folder)

        for image_folder_path in input_image_folder_paths:
            os.rename(image_folder_path, os.path.join(
                self.DATA_ROOT, Path(image_folder_path).stem))
