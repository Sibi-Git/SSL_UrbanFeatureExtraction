import dota_utils as util
import os
import cv2
import json

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def DOTA2COCO():
    srcpath = '/Users/deeptisaravanan/Downloads/dota_old/train/'
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    #data_dict['annotations'] = []
    #for idex, name in enumerate(wordname_15):
        #single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        #data_dict['labels'] = []

    inst_count = 1
    image_id = 1
    filenames = util.GetFileFromThisRootDir(labelparent)
    #print(labelparent)
    count=1
    for file in filenames:
        with open('/Users/deeptisaravanan/Downloads/traindotaAnnotation/' + str(count) + '.json', 'w') as f_out:
            data_dict = {}
            #data_dict['image_size'] = []
            data_dict['bboxes'] = []
            data_dict['labels'] = []
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            if('DS_Store' in basename + '.png'):
                print(basename + '.png')
            else:
                #print(imagepath)
                img = cv2.imread(imagepath)
                #print(imagepath)
                height, width, c = img.shape

                #single_image = {}
                #single_image['file_name'] = basename + '.png'
                #single_image['id'] = image_id
                #single_image['width'] = width
                #single_image['height'] = height
                #single_image['image_size'] = width, height
                data_dict['image_size'] = [width, height]

                # annotations
                objects = util.parse_dota_poly2(file)
                for obj in objects:
                    #single_obj = {}
                    #single_obj['area'] = obj['area']
                    #single_obj['category_id'] = wordname_15.index(obj['name']) + 1
                    #single_obj['segmentation'] = []
                    #single_obj['segmentation'].append(obj['poly'])
                    #single_obj['iscrowd'] = 0
                    xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                            max(obj['poly'][0::2]), max(obj['poly'][1::2])

                    width, height = xmax - xmin, ymax - ymin
                    #single_obj['bbox'] = xmin, ymin, width, height
                    #single_obj['image_id'] = image_id
                    #data_dict['bboxes'].append([xmin, ymin, width, height])
                    data_dict['bboxes'].append([xmin, ymin, xmax, ymax])
                    data_dict['labels'].append(obj['name'])
                    #single_obj['id'] = inst_count
                    inst_count = inst_count + 1
                image_id = image_id + 1
                json.dump(data_dict, f_out)
                #print(imagepath)
                #print(os.path.join(imageparent, str(count) + '.png'))
                os.rename(imagepath, os.path.join(imageparent, str(count) + '.png'))
                count+=1
    print(count-1)

if __name__ == '__main__':
    #DOTA2COCO(r'/Users/deeptisaravanan/Desktop/Fall_2022/Computer_Vision/Project/dota/val/', r'/Users/deeptisaravanan/Desktop/Fall_2022/Computer_Vision/Project/dota/annotations/instances_val.json')
    DOTA2COCO()
