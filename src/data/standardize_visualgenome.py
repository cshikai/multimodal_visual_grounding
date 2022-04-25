
import pandas as pd
import os
import json

from sklearn.model_selection import train_test_split
# images_info = json.load(f)['images']

# for image_info in images_info:
#     filename = 'mscoco/' + image['file_name']


class VgStandardizer():
    WORD_LIMIT = 20

    def __init__(self, max_captions=5, valid_size=0.2):
        self.max_captions = max_captions
        self.valid_size = valid_size
        self.seed = 42

    def word_count(self, string):
        return len(string.split(" "))

    def filter_long_captions(self, manifest):

        manifest['word_count'] = manifest.caption.apply(self.word_count)
        manifest = manifest[manifest.word_count <= self.WORD_LIMIT]\
            .drop(columns=['word_count'])
        return manifest

    def standardize(self):
        manifest = []
        with open('/data/raw/region_descriptions.json') as f:
            original_manifest = json.load(f)

            for i in original_manifest:
                single_image_manifest = {
                    'filename': [], 'caption': [], 'size': []}
                filename = 'visualgenome/'+str(i['id'])+'.jpg'

                for k in i['regions']:
                    single_image_manifest['filename'].append(filename)
                    phrase = k['phrase']
                    phrase = phrase[0].upper(
                    ) + phrase[1:].lower()
                    phrase = phrase.strip(' ').strip(
                        '.').strip(' ').strip('\n')+'.'
                    single_image_manifest['caption'].append(phrase)
                    single_image_manifest['size'].append(
                        k['width']*k['height'])

                single_image_manifest = pd.DataFrame(single_image_manifest)

                single_image_manifest = single_image_manifest.sort_values(
                    by=['size'], ascending=False).head(self.max_captions).drop(columns=['size'])

                manifest.append(single_image_manifest)

        manifest = pd.concat(manifest, ignore_index=True)
        manifest = self.filter_long_captions(manifest)
        manifest_train, manifest_valid = self._train_test_split(manifest)

        manifest_train.to_csv(
            '/data/manifests/visualgenome/train_manifest.csv', index=False)
        manifest_valid.to_csv(
            '/data/manifests/visualgenome/valid_manifest.csv', index=False)

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
    std = VgStandardizer()
    std.standardize()
