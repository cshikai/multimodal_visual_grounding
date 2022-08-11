
import os
from typing import List
import pandas as pd
import dask.dataframe as dd


class ImageManifestCreator():

    def __init__(self, npartitions):
        self.npartitions = npartitions

    def create(self, list_of_manifest: List[str], destination_folder: str) -> None:
        '''
        Even though there may be multiple captions for each image,
        there should be only one caption per image for each batch.
        i.e captions in a batch should only be valid for one image only. 
        '''

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        manifest_df = []
        for manifest in list_of_manifest:
            single_manifest_df = pd.read_csv(manifest)
            manifest_df.append(single_manifest_df)
        manifest_df = pd.concat(manifest_df, ignore_index=True)

        manifest_df = manifest_df.drop_duplicates(subset=['filename'])
        print('Creating Dask Dataset')
        print(len(manifest_df))
        manifest_df = manifest_df.reset_index(drop=True)
        dask_df = dd.from_pandas(manifest_df, npartitions=self.npartitions)

        # dask_df = dask_df.set_index('batch_number', sorted=True)
        # only fastparquet preserve categoricals
        print('Saving Dask Dataset')
        dask_df.to_parquet(os.path.join(
            destination_folder, 'data.parquet'), engine='fastparquet')


if __name__ == '__main__':
    dc = ImageManifestCreator(npartitions=100)
    dc.create(['/data/manifests/flickr/valid_manifest.csv', '/data/manifests/mscoco/valid_manifest.csv', '/data/manifests/visualgenome/valid_manifest.csv'],
              '/data/image_manifest/flickr_mscoco_visualgenome/valid')
    dc.create(['/data/manifests/flickr/train_manifest.csv', '/data/manifests/mscoco/train_manifest.csv', '/data/manifests/visualgenome/train_manifest.csv'],
              '/data/image_manifest/flickr_mscoco_visualgenome/train')
