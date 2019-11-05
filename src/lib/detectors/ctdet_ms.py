from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from external.nms import soft_nms
from models.decode import ctdet_decode, ctdet_decode_ms
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process, ctdet_post_process_ms
from utils.debugger import Debugger
import torch.nn.functional as F

from .base_detector_ms import BaseDetector


class CtdetDetector_ms(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector_ms, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            hm_s8 = output['hm_s8'].sigmoid_()
            wh_s8 = output['wh_s8']
            reg_s8 = output['reg_s8'] if self.opt.reg_offset else None
            hm_s16 = output['hm_s16'].sigmoid_()
            wh_s16 = output['wh_s16']
            reg_s16 = output['reg_s16'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
                hm_s8 = (hm_s8[0:1] + flip_tensor(hm_s8[1:2])) / 2
                wh_s8 = (wh_s8[0:1] + flip_tensor(wh_s8[1:2])) / 2
                reg_s8 = reg_s8[0:1] if reg_s8 is not None else None
                hm_s16 = (hm_s16[0:1] + flip_tensor(hm_s16[1:2])) / 2
                wh_s16 = (wh_s16[0:1] + flip_tensor(wh_s16[1:2])) / 2
                reg_s16 = reg_s16[0:1] if reg_s16 is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()

            # wh_s8_up = F.upsample(wh_s8, scale_factor=2, mode='nearest')
            # wh_s16_up = F.upsample(wh_s16, scale_factor=4, mode='nearest')

            # h_s4 = wh[:, 1, :, :]
            # w_s4 = wh[:, 0, :, :]
            # aspect_s4 = h_s4 * w_s4
            # # fea_ct, feat_h, feat_w = h_s4.size()
            # ind_s4 = torch.where(aspect_s4 < 32*32, torch.full_like(aspect_s4, 1), torch.full_like(aspect_s4, 0)).unsqueeze(0)
            # ind_s8 = torch.where(aspect_s4 >= 32*32, torch.full_like(aspect_s4, 1), torch.full_like(aspect_s4, 0)).unsqueeze(0)
            # # ind_s16 = torch.where(aspect_s4 > 32*32, torch.full_like(h_s4, 1), torch.full_like(h_s4, 0)).unsqueeze(0)
            # # ind_all = torch.ones_like(h_s4).unsqueeze(0)
            # # ind_s8 = ind_all - ind_s4 -ind_s16
            # #
            # # wh_new = wh*ind_s4 + wh_s8_up*ind_s8*2 + wh_s16_up*ind_s16*4
            # wh_new = wh*ind_s4 + wh_s8_up*ind_s8*2

            # dets_s4 = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)
            # dets_s8 = ctdet_decode(hm_s8, wh_s8, reg=reg_s8, K=self.opt.K)
            # dets_s16 = ctdet_decode(hm_s16, wh_s16, reg=reg_s16, K=self.opt.K)

            dets_s4 = ctdet_decode(hm, wh, reg=reg, K=40)
            dets_s8 = ctdet_decode(hm_s8, wh_s8, reg=reg_s8, K=30)
            dets_s16 = ctdet_decode(hm_s16, wh_s16, reg=reg_s16, K=30)

            # dets_s4 = ctdet_decode_ms(hm, wh, reg=reg, K=self.opt.K)

        if return_time:
            # return output, dets_s4, dets_s8, forward_time
            return output, dets_s4, dets_s8, dets_s16, forward_time
        else:
            # return output, dets_s4, dets_s8
            return output, dets_s4, dets_s8, dets_s16

    def post_process(self, dets, dets_s8, dets_s16, meta, scale=1):
        # def post_process(self, dets, dets_s8, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets_s8 = dets_s8.detach().cpu().numpy()
        dets_s8 = dets_s8.reshape(1, -1, dets_s8.shape[2])

        dets_s16 = dets_s16.detach().cpu().numpy()
        dets_s16 = dets_s16.reshape(1, -1, dets_s16.shape[2])
        # dets_total = ctdet_post_process(
        #     dets.copy(), [meta['c']], [meta['s']],
        #     meta['out_height'], meta['out_width'], self.opt.num_classes)
        # dets_total = ctdet_post_process(
        #   dets_s8.copy(), [meta['c']], [meta['s']],
        #   meta['out_height_s8'], meta['out_width_s8'], self.opt.num_classes)

        # dets_total = ctdet_post_process(
        #  dets_s16.copy(), [meta['c']], [meta['s']],
        #  meta['out_height_s16'], meta['out_width_s16'], self.opt.num_classes)

        dets_total = ctdet_post_process_ms(
            dets.copy(), dets_s8.copy(), dets_s16.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], meta['out_height_s8'], meta['out_width_s8'],
            meta['out_height_s16'], meta['out_width_s16'],
            self.opt.num_classes)

        for j in range(1, self.num_classes + 1):
            dets_total[0][j] = np.array(dets_total[0][j], dtype=np.float32).reshape(-1, 5)
            dets_total[0][j][:, :4] /= scale
        return dets_total[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            # ---------s4, s8 -----------------
            # soft_nms(results[j], Nt=0.5, method=2)
            # ------------------------------------
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)