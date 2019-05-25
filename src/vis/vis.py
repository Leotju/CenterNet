import numpy as np
import cv2

# idx = [1, 3, 6, 9, 12]
# idx = [6, 13, 23, 33, 43]
idx = [5, 8, 10, 13, 16]
for i in idx:
    feats = np.load('/home/leo/Pictures/3/frn/' + str(i) + '.npy')
    # for j in range(10):
    # print(feats.shape)
    feats_max = feats[0, :, :, :].max(0)
        # feats_max = feats[0, j, :, :]
    feats_norm = feats_max / feats.max() * 255
    feats_norm = feats_norm.astype(np.uint8)
    feats_norm = cv2.applyColorMap(feats_norm, cv2.COLORMAP_JET)
    cv2.imwrite('/home/leo/Pictures/3/frn/' + str(i) + '.png', feats_norm)
