from typing import Dict, Tuple
import os

from torch.utils.data import Dataset
import dask.dataframe as dd
from PIL import Image

from .preprocessor import PreProcessor


class VisualGroundingDataset(Dataset):
    """
    """
    DATA_ROOT = '/data'

    def __init__(self, mode: str, cfg: Dict) -> None:
        """
        """
        self.root_folder = os.path.join(self.DATA_ROOT, mode)
        self.data = dd.read_parquet(os.path.join(self.root_folder, 'data.parquet'),
                                    columns=['filename', 'caption'],
                                    engine='fastparquet')  # this is lazy loading, its not actually loading into memory

        self.preprocessor = PreProcessor(cfg)

    def __len__(self) -> int:
        '''
        Get the length of dataset.
        '''
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Image.Image, str]:
        '''
        Get the item for each batch
        :return: a tuple of 6 object:
        1) normalized features of dataset
        2) labels of dataset (one-hot encoded and labels_dct)
        '''
        data_slice = self.data.loc[index].compute()

        # data values loaded are between 0 and 255, with the shape [h,w,c], c is the number of channels , usually RGB. both PIL and skimages will achieve this
        image = Image.open(os.path.join(
            self.DATA_ROOT, data_slice['filename'].values[0]))

        text = data_slice['caption'].values[0]

        image, text, length = self.preprocessor((image, text))

        return image, text, length
