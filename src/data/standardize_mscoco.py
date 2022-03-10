
import pandas as pd
import os
import json
# images_info = json.load(f)['images']

# for image_info in images_info:
#     filename = 'mscoco/' + image['file_name']


class MscocoStandardizer():

    def standardize(self, input_manifest, output_manifest):
        # os.rename('/data/flickr30k-images', '/data/flickr')
        manifest = {'filename': [], 'caption': []}

        with open(input_manifest) as f:
            id2file = {}
            no_map = []
            original_manifest = json.load(f)

            for image_info in original_manifest['images']:
                id2file[image_info['id']] = 'mscoco/' + image_info['file_name']

            for annotation_info in original_manifest['annotations']:

                # if annotation_info['id'] in id2file:
                manifest['filename'].append(
                    id2file[annotation_info['image_id']])
                manifest['caption'].append(annotation_info['caption'])
                # else:
                #     no_map.append(annotation_info)
        # print(len(id2file))
        # print(len(no_map))
        # print(manifest)
        # print(manifest)
        #     for line in f:
        #         filename_captionnum, caption = line.split('\t')
        #         filename, caption_num = filename_captionnum.split('#')

        #         manifest['filename'].append('flickr/'+filename)
        #         manifest['caption'].append(
        #             caption.strip(' .\n').strip('''"'''))

        #         # break
        manifest = pd.DataFrame(manifest)
        manifest.to_csv(output_manifest, index=False)


if __name__ == '__main__':
    std = MscocoStandardizer()
    std.standardize('/data/annotations/captions_train2014.json',
                    '/data/train_manifest_mscoco.csv')
    std.standardize('/data/annotations/captions_val2014.json',
                    '/data/valid_manifest_mscoco.csv')
