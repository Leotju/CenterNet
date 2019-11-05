from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.ctdet_ms import CTDetDataset_ms
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.pascal07 import PascalVOC07
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.coco2014 import COCO2014
from .dataset.coco_tiny import COCOTINY
dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'pascal07': PascalVOC07,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'coco2014': COCO2014,
  'coco_tiny': COCOTINY
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ctdet_ms': CTDetDataset_ms,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
