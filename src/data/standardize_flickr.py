
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class FlickrStandardizer():

    def __init__(self, valid_size=0.2):
        self.valid_size = valid_size
        self.seed = 42

    def standardize(self, input_manifest, output_manifest):
        # os.rename('/data/flickr30k-images', '/data/flickr')
        manifest = {'filename': [], 'caption': []}
        with open(input_manifest) as f:
            for line in f:
                filename_captionnum, caption = line.split('\t')
                filename, caption_num = filename_captionnum.split('#')

                manifest['filename'].append('flickr/'+filename)
                manifest['caption'].append(
                    caption.strip(' .\n').strip('''"''')+'.')

                # break
        manifest = pd.DataFrame(manifest)
        manifest_train, manifest_valid = self._train_test_split(manifest)

        manifest_train.to_csv('/data/train_manifest.csv', index=False)
        manifest_valid.to_csv('/data/valid_manifest.csv', index=False)

    def _train_test_split(self, df):
        df['file_id'] = df.groupby('filename').ngroup()
        df.index = df.file_id
        df_images = df.groupby('filename').first().set_index('file_id')
        track_train, track_valid = train_test_split(
            df_images, test_size=self.valid_size, random_state=self.seed)
        df.loc[track_train.index, 'is_train'] = True
        df.loc[track_valid.index, 'is_train'] = False
        df.is_train = df.is_train.astype(bool)

        df_train = df[df['is_train']].drop(
            columns=['is_train', 'file_id']).reset_index(drop=True)
        df_valid = df[~df['is_train']].drop(
            columns=['is_train', 'file_id']).reset_index(drop=True)
        return df_train,  df_valid


if __name__ == '__main__':
    std = FlickrStandardizer()
    std.standardize('/data/results_20130124.token', '/data/manifest.csv')
    # std.standardize('/data/results_20130124.token', '/data/test_manifest.csv')
