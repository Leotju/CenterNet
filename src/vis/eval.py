import json
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

def convert(json_result, ann_json):
    dets = []
    for cat_id, jr in enumerate(json_result):
        for im_id, j in enumerate(jr):
            if len(j) > 0:
                for ji in j:
                    det = {}
                    det['image_id'] = ann_json['images'][im_id]['id']
                    det['category_id'] = cat_id
                    det['score'] = ji[4]
                    det['bbox'] = [ji[0], ji[1], ji[2] - ji[0] +1 , ji[3] - ji[1] + 1]
                    dets.append(det)
    return dets




ann_path = '/media/leo/data/code/pangnet/detection/object_as_point/Git/CenterNet/data/voc/annotations/pascal_test2007.json'
# pred_path = '/media/leo/data/code/pangnet/detection/object_as_point/Git/CenterNet/exp/ctdet/fatnet_frn_pascal_192_daspp_dcn_dla/results.json'
# pred_path = '/media/leo/data/code/pangnet/detection/object_as_point/Git/CenterNet/exp/ctdet/resnet_18_dcn_lr_1x/results.json'
pred_path = '/media/leo/data/code/pangnet/detection/object_as_point/Git/CenterNet/exp/ctdet/fatnet_frn_pascal_192_93/results.json'
ann_json = json.load(open(ann_path))
pred = json.load(open(pred_path))
jcoco = json.load(open('/media/leo/data/code/mm_ped/work_dirs/faster_rcnn_r50_fpn_baseline/12.json'))

pred_convert = convert(pred, ann_json)

pred_convert_path = pred_path[:-4] + '_convert.json'
json.dump(pred_convert, open(pred_convert_path, 'w'))

coco = coco.COCO(ann_path)
dets = coco.loadRes(pred_convert_path)
img_ids = coco.getImgIds()
num_images = len(img_ids)
coco_eval = COCOeval(coco, dets, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()