from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math


class CTDetDataset_ms(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        # output_h = [input_h // self.opt.down_ratio[0], input_h // self.opt.down_ratio[1], input_h // self.opt.down_ratio[2],input_h // self.opt.down_ratio[3]]
        # output_w = [input_w // self.opt.down_ratio[0], input_w // self.opt.down_ratio[1], input_w // self.opt.down_ratio[2],input_w // self.opt.down_ratio[3]]
        output_h = [input_h // self.opt.down_ratio, input_h // self.opt.down_ratio,
                    input_h // self.opt.down_ratio]
        output_w = [input_w // self.opt.down_ratio, input_w // self.opt.down_ratio,
                    input_w // self.opt.down_ratio]

        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w[0], output_h[0]])
        trans_output_s8 = get_affine_transform(c, s, 0, [output_w[1], output_h[1]])
        trans_output_s16 = get_affine_transform(c, s, 0, [output_w[2], output_h[2]])
        # trans_output_s32 = get_affine_transform(c, s, 0, [output_w[3], output_h[3]])

        hm = np.zeros((num_classes, output_h[0], output_w[0]), dtype=np.float32)
        hm_s8 = np.zeros((num_classes, output_h[1], output_w[1]), dtype=np.float32)
        hm_s16 = np.zeros((num_classes, output_h[2], output_w[2]), dtype=np.float32)
        # hm_s32 = np.zeros((num_classes, output_h[3], output_w[3]), dtype=np.float32)

        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        wh_s8 = np.zeros((self.max_objs, 2), dtype=np.float32)
        wh_s16 = np.zeros((self.max_objs, 2), dtype=np.float32)
        # wh_s32 = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h[0], output_w[0]), dtype=np.float32)
        dense_wh_s8 = np.zeros((2, output_h[1], output_w[1]), dtype=np.float32)
        dense_wh_s16 = np.zeros((2, output_h[2], output_w[2]), dtype=np.float32)
        # dense_wh_s32 = np.zeros((2, output_h[3], output_w[3]), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_s8 = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_s16 = np.zeros((self.max_objs, 2), dtype=np.float32)
        # reg_s32 = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        ind_s8 = np.zeros((self.max_objs), dtype=np.int64)
        ind_s16 = np.zeros((self.max_objs), dtype=np.int64)
        # ind_s32 = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        reg_mask_s8 = np.zeros((self.max_objs), dtype=np.uint8)
        reg_mask_s16 = np.zeros((self.max_objs), dtype=np.uint8)
        # reg_mask_s32 = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_wh_s8 = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_wh_s16 = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        # cat_spec_wh_s32 = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        cat_spec_mask_s8 = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        cat_spec_mask_s16 = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        # cat_spec_mask_s32 = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        gt_det_s8 = []
        gt_det_s16 = []
        # gt_det_s32 = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox_s8 = bbox.copy()
            bbox_s16 = bbox.copy()
            # bbox_s32 = bbox.copy()

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w[0] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h[0] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            bbox_s8[:2] = affine_transform(bbox_s8[:2], trans_output_s8)
            bbox_s8[2:] = affine_transform(bbox_s8[2:], trans_output_s8)
            bbox_s8[[0, 2]] = np.clip(bbox_s8[[0, 2]], 0, output_w[1] - 1)
            bbox_s8[[1, 3]] = np.clip(bbox_s8[[1, 3]], 0, output_h[1] - 1)
            h_s8, w_s8 = bbox_s8[3] - bbox_s8[1], bbox_s8[2] - bbox_s8[0]

            bbox_s16[:2] = affine_transform(bbox_s16[:2], trans_output_s16)
            bbox_s16[2:] = affine_transform(bbox_s16[2:], trans_output_s16)
            bbox_s16[[0, 2]] = np.clip(bbox_s16[[0, 2]], 0, output_w[2] - 1)
            bbox_s16[[1, 3]] = np.clip(bbox_s16[[1, 3]], 0, output_h[2] - 1)
            h_s16, w_s16 = bbox_s16[3] - bbox_s16[1], bbox_s16[2] - bbox_s16[0]

            # bbox_s32[:2] = affine_transform(bbox_s32[:2], trans_output_s32)
            # bbox_s32[2:] = affine_transform(bbox_s32[2:], trans_output_s32)
            # bbox_s32[[0, 2]] = np.clip(bbox_s32[[0, 2]], 0, output_w[3] - 1)
            # bbox_s32[[1, 3]] = np.clip(bbox_s32[[1, 3]], 0, output_h[3] - 1)
            # h_s32, w_s32 = bbox_s32[3] - bbox_s32[1], bbox_s32[2] - bbox_s32[0]

            # if h > 0 and w > 0:
            b_aspect = h * w
            if b_aspect > 0 and b_aspect < (16 * 16):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w[0] + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

            elif b_aspect >= (16 * 16) and b_aspect < (32 * 32):
                # ----------------s8----------------------
                radius_s8 = gaussian_radius((math.ceil(h_s8), math.ceil(w_s8)))
                radius_s8 = max(0, int(radius_s8))
                radius_s8 = self.opt.hm_gauss if self.opt.mse_loss else radius_s8
                ct_s8 = np.array(
                    [(bbox_s8[0] + bbox_s8[2]) / 2, (bbox_s8[1] + bbox_s8[3]) / 2], dtype=np.float32)
                ct_int_s8 = ct_s8.astype(np.int32)
                draw_gaussian(hm_s8[cls_id], ct_int_s8, radius_s8)
                wh_s8[k] = 1. * w_s8, 1. * h_s8
                ind_s8[k] = ct_int_s8[1] * output_w[1] + ct_int_s8[0]
                reg_s8[k] = ct_s8 - ct_int_s8
                reg_mask_s8[k] = 1
                cat_spec_wh_s8[k, cls_id * 2: cls_id * 2 + 2] = wh_s8[k]
                cat_spec_mask_s8[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh_s8, hm_s8.max(axis=0), ct_int_s8, wh_s8[k], radius_s8)
                gt_det_s8.append([ct_s8[0] - w_s8 / 2, ct_s8[1] - h_s8 / 2,
                                  ct_s8[0] + w_s8 / 2, ct_s8[1] + h_s8 / 2, 1, cls_id])

            # elif b_aspect >= (32 * 32):
            else:
                # ----------------s16----------------------
                radius_s16 = gaussian_radius((math.ceil(h_s16), math.ceil(w_s16)))
                radius_s16 = max(0, int(radius_s16))
                radius_s16 = self.opt.hm_gauss if self.opt.mse_loss else radius_s16
                ct_s16 = np.array(
                    [(bbox_s16[0] + bbox_s16[2]) / 2, (bbox_s16[1] + bbox_s16[3]) / 2], dtype=np.float32)
                ct_int_s16 = ct_s16.astype(np.int32)
                draw_gaussian(hm_s16[cls_id], ct_int_s16, radius_s16)
                wh_s16[k] = 1. * w_s16, 1. * h_s16
                ind_s16[k] = ct_int_s16[1] * output_w[2] + ct_int_s16[0]
                reg_s16[k] = ct_s16 - ct_int_s16
                reg_mask_s16[k] = 1
                cat_spec_wh_s16[k, cls_id * 2: cls_id * 2 + 2] = wh_s16[k]
                cat_spec_mask_s16[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh_s16, hm_s16.max(axis=0), ct_int_s16, wh_s16[k], radius_s16)
                gt_det_s16.append([ct_s16[0] - w_s16 / 2, ct_s16[1] - h_s16 / 2,
                                   ct_s16[0] + w_s16 / 2, ct_s16[1] + h_s16 / 2, 1, cls_id])

                # ----------------s32----------------------
                # radius_s32 = gaussian_radius((math.ceil(h_s32), math.ceil(w_s32)))
                # radius_s32 = max(0, int(radius_s32))
                # radius_s32 = self.opt.hm_gauss if self.opt.mse_loss else radius_s32
                # ct_s32 = np.array(
                #   [(bbox_s32[0] + bbox_s32[2]) / 2, (bbox_s32[1] + bbox_s32[3]) / 2], dtype=np.float32)
                # ct_int_s32 = ct_s32.astype(np.int32)
                # draw_gaussian(hm_s32[cls_id], ct_int_s32, radius_s32)
                # wh_s32[k] = 1. * w_s32, 1. * h_s32
                # ind_s32[k] = ct_int_s32[1] * output_w[2] + ct_int_s32[0]
                # reg_s32[k] = ct_s32 - ct_int_s32
                # reg_mask_s32[k] = 1
                # cat_spec_wh_s32[k, cls_id * 2: cls_id * 2 + 2] = wh_s32[k]
                # cat_spec_mask_s32[k, cls_id * 2: cls_id * 2 + 2] = 1
                # if self.opt.dense_wh:
                #   draw_dense_reg(dense_wh_s32, hm_s32.max(axis=0), ct_int_s32, wh_s32[k], radius_s32)
                # gt_det_s32.append([ct_s32[0] - w_s32 / 2, ct_s32[1] - h_s32 / 2,
                #                    ct_s32[0] + w_s32 / 2, ct_s32[1] + h_s32 / 2, 1, cls_id])

        # ret = {'input': inp, 'hm': hm,'hm_s8': hm_s8, 'hm_s16': hm_s16, 'hm_s32': hm_s32, 'reg_mask': reg_mask, 'reg_mask_s8': reg_mask_s8, 'reg_mask_s16': reg_mask_s16,
        #        'reg_mask_s32': reg_mask_s32, 'ind': ind, 'ind_s8': ind_s8, 'ind_s16': ind_s16, 'ind_s32': ind_s32, 'wh': wh, 'wh_s8': wh_s8, 'wh_s16': wh_s16,'wh_s32': wh_s32}
        ret = {'input': inp, 'hm_s': hm, 'hm_m': hm_s8, 'hm_l': hm_s16, 'reg_mask_s': reg_mask,
               'reg_mask_m': reg_mask_s8, 'reg_mask_l': reg_mask_s16,
               'ind_s': ind, 'ind_m': ind_s8, 'ind_l': ind_s16, 'wh_s': wh, 'wh_m': wh_s8, 'wh_l': wh_s16}

        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']

            hm_a_s8 = hm_s8.max(axis=0, keepdims=True)
            dense_wh_mask_s8 = np.concatenate([hm_a_s8, hm_a_s8], axis=0)
            ret.update({'dense_wh_s8': dense_wh_s8, 'dense_wh_mask_s8': dense_wh_mask_s8})
            del ret['wh_s8']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
            ret.update({'cat_spec_wh_s8': cat_spec_wh_s8, 'cat_spec_mask_s8': cat_spec_mask_s8})
            del ret['wh_s8']
        if self.opt.reg_offset:
            ret.update({'reg_s': reg})
            ret.update({'reg_m': reg_s8})
            ret.update({'reg_l': reg_s16})
            # ret.update({'reg_s32': reg_s32})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            gt_det_s8 = np.array(gt_det_s8, dtype=np.float32) if len(gt_det_s8) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            gt_det_s16 = np.array(gt_det_s8, dtype=np.float32) if len(gt_det_s16) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'gt_det_s8': gt_det_s8, 'gt_det_s16': gt_det_s16,
                    'img_id': img_id}
            # 'gt_det_s32': gt_det_s32, 'img_id': img_id}
            ret['meta'] = meta
        return ret
