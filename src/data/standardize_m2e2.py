
import pandas as pd
import os
import json
from tqdm import tqdm
# images_info = json.load(f)['images']

# for image_info in images_info:
#     filename = 'mscoco/' + image['file_name']


class M2E2Standardizer():

    WORD_LIMIT = 40

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
            original_manifest = json.load(f)

            for articles in tqdm(original_manifest):

                for img in original_manifest[articles]:
                    manifest['filename'].append(
                        "images_m2e2/{}_{}.jpg".format(articles, img))
                    caption = original_manifest[articles][img]['caption']
                    if caption[:7] == 'FILE - ':  # Strip the extra thingie infront
                        caption = caption[7:]
                    caption = caption.replace("&#39;", "'")
                    caption = caption.replace("&nbsp;", " ")
                    caption = caption.replace("&amp;", "&")

                    manifest['caption'].append(caption)
        print(len(manifest['filename']))
        manifest = pd.DataFrame(manifest)
        manifest['caption'] = manifest['caption'].apply(
            lambda x: x.strip(' ').strip('.').strip(' ').strip('\n')+'.')
        manifest = self.filter_long_captions(manifest)

        manifest.to_csv(output_manifest, index=False)


if __name__ == '__main__':
    std = M2E2Standardizer()
    std.standardize('/data/raw/m2e2_train_caption.json',
                    '/data/manifests/m2e2/train_manifest.csv')
    std.standardize('/data/raw/m2e2_valid_caption.json',
                    '/data/manifests/m2e2/valid_manifest.csv')
