
import pandas as pd
import os
import json
# images_info = json.load(f)['images']

# for image_info in images_info:
#     filename = 'mscoco/' + image['file_name']


class MscocoStandardizer():

    WORD_LIMIT = 20

    def word_count(self, string):
        return len(string.split(" "))

    def filter_long_captions(self, manifest):

        manifest['word_count'] = manifest.caption.apply(self.word_count)
        manifest = manifest[manifest.word_count <= self.WORD_LIMIT]\
            .drop(columns=['word_count'])
        return manifest

    def standardize(self, input_manifest, output_manifest):
        manifest = {'filename': [], 'caption': []}

        with open(input_manifest) as f:
            id2file = {}
            original_manifest = json.load(f)

            for image_info in original_manifest['images']:
                id2file[image_info['id']] = 'mscoco/' + image_info['file_name']

            for annotation_info in original_manifest['annotations']:

                # if annotation_info['id'] in id2file:
                manifest['filename'].append(
                    id2file[annotation_info['image_id']])
                manifest['caption'].append(annotation_info['caption'])

        manifest = pd.DataFrame(manifest)
        manifest['caption'] = manifest['caption'].apply(
            lambda x: x.strip(' ').strip('.').strip(' ').strip('\n')+'.')
        manifest = self.filter_long_captions(manifest)

        manifest.to_csv(output_manifest, index=False)


if __name__ == '__main__':
    std = MscocoStandardizer()
    std.standardize('/data/raw/captions_train2014.json',
                    '/data/manifests/mscoco/train_manifest.csv')
    std.standardize('/data/raw/captions_val2014.json',
                    '/data/manifests/mscoco/valid_manifest.csv')
