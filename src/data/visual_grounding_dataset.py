
import os
from typing import List
import pandas as pd
import dask.dataframe as dd


class VisualGroundingDataCreator():

    def __init__(self, batch_size: int = 32, num_captions: int = 5, nparitions: int = 100):
        self.batch_size = batch_size
        self.num_captions = num_captions
        self.npartitions = nparitions

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

        print('Extracting the top {} captions'.format(self.num_captions))
        manifest_df = manifest_df.groupby(
            'filename').apply(lambda x: x.head(self.num_captions)).reset_index(drop=True)

        print('Assigning image to a cluster')
        manifest_df['cluster_number'] = manifest_df.groupby(
            'filename').ngroup() // self.batch_size

        print('Assigning caption number for each image')
        manifest_df['intra_group_number'] = manifest_df.groupby(
            'filename')['cluster_number'].rank('first')

        print('Assigning batch number based on cluster and caption number')
        manifest_df['batch_number'] = manifest_df.groupby(
            ['cluster_number', 'intra_group_number']).ngroup()

        print('Sorting dataset based on batch number')
        manifest_df = manifest_df.sort_values(
            by=['batch_number'], ascending=True)\
            .reset_index(drop=True)\
            .drop(columns=['cluster_number', 'intra_group_number'])

        print('Creating Dask Dataset')
        print('len of df: ', len(manifest_df))
        dask_df = dd.from_pandas(manifest_df, npartitions=self.npartitions)

        # dask_df = dask_df.set_index('batch_number', sorted=True)
        # only fastparquet preserve categoricals
        print('Saving Dask Dataset')
        dask_df.to_parquet(os.path.join(
            destination_folder, 'data.parquet'), engine='fastparquet')


if __name__ == '__main__':
    dc = VisualGroundingDataCreator(num_captions=5, nparitions=100)
    dc.create(['/data/manifest/flickr/valid_manifest.csv'], '/data/valid')
    dc.create(['/data/manifest/flickr/train_manifest.csv'], '/data/train')
