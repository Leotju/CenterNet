import numpy as np
import cv2

feats = np.load('/media/leo/data/code/pangnet/detection/object_as_point/Git/CenterNet/src/1.npy')

feats_max = feats[0,:,:,:].mean(0)

feats_norm = feats_max / feats.max() * 255

feats_norm = feats_norm.astype(np.uint8)
# cv2.cvtColor()

cv2.imwrite('mean.png', feats_norm)