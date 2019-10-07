from pycocotools.coco import COCO
import json, os

coco_root = './data/coco/annotations'
ann = COCO(os.path.join(coco_root, 'instances_train2017.json'))
ann_js = json.load(open(os.path.join(coco_root, 'instances_train2017.json')))
filter_boxes = []
filter_imgs = []
for img_id, img in enumerate(ann.imgs):
    if img_id < 5000:
        filter_imgs.append(ann.imgs[img])
        boxes = ann.imgToAnns[img_id]
        for box in boxes:
            filter_boxes.append(box)
ann_js['images'] = filter_imgs
ann_js['annotations'] = filter_boxes

json.dump(ann_js, open(os.path.join(coco_root, 'instances_tiny_train2017.json', 'w')))
