from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .ctdet_ms import CtdetTrainer_ms

train_factory = {
    'exdet': ExdetTrainer,
    'ddd': DddTrainer,
    'ctdet': CtdetTrainer,
    'multi_pose': MultiPoseTrainer,
    'ctdet_ms': CtdetTrainer_ms,
}
