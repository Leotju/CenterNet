from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_s_loss, wh_s_loss, off_s_loss = 0, 0, 0
        hm_m_loss, wh_m_loss, off_m_loss = 0, 0, 0
        hm_l_loss, wh_l_loss, off_l_loss = 0, 0, 0
        # hm_s32_loss, wh_s32_loss, off_s32_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm_s'] = _sigmoid(output['hm_s'])
                output['hm_m'] = _sigmoid(output['hm_m'])
                output['hm_l'] = _sigmoid(output['hm_l'])
                # output['hm_s32'] = _sigmoid(output['hm_s32'])

            if opt.eval_oracle_hm:
                output['hm_S'] = batch['hm_s']
                output['hm_m'] = batch['hm_m']
                output['hm_l'] = batch['hm_l']
                # output['hm_s32'] = batch['hm_s32']
            if opt.eval_oracle_wh:
                output['wh_s'] = torch.from_numpy(gen_oracle_map(
                    batch['wh_s'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh_s'].shape[3], output['wh_s'].shape[2])).to(opt.device)
                output['wh_m'] = torch.from_numpy(gen_oracle_map(
                    batch['wh_m'].detach().cpu().numpy(),
                    batch['ind_m'].detach().cpu().numpy(),
                    output['wh_m'].shape[3], output['wh_m'].shape[2])).to(opt.device)
                output['wh_l'] = torch.from_numpy(gen_oracle_map(
                    batch['wh_l'].detach().cpu().numpy(),
                    batch['ind_l'].detach().cpu().numpy(),
                    output['wh_l'].shape[3], output['wh_l'].shape[2])).to(opt.device)
                # output['wh_s32'] = torch.from_numpy(gen_oracle_map(
                #   batch['wh_s32'].detach().cpu().numpy(),
                #   batch['ind_s32'].detach().cpu().numpy(),
                #   output['wh_s32'].shape[3], output['wh_s32'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg_s'].detach().cpu().numpy(),
                    batch['ind_s'].detach().cpu().numpy(),
                    output['reg_s'].shape[3], output['reg_s'].shape[2])).to(opt.device)
                output['reg_m'] = torch.from_numpy(gen_oracle_map(
                    batch['reg_m'].detach().cpu().numpy(),
                    batch['ind_m'].detach().cpu().numpy(),
                    output['reg_m'].shape[3], output['reg_m'].shape[2])).to(opt.device)
                output['reg_s16'] = torch.from_numpy(gen_oracle_map(
                    batch['reg_l'].detach().cpu().numpy(),
                    batch['ind_l'].detach().cpu().numpy(),
                    output['reg_l'].shape[3], output['reg_l'].shape[2])).to(opt.device)


            hm_s_loss += self.crit(output['hm_s'], batch['hm_s']) / opt.num_stacks
            hm_m_loss += self.crit(output['hm_m'], batch['hm_m']) / opt.num_stacks
            hm_l_loss += self.crit(output['hm_l'], batch['hm_l']) / opt.num_stacks
            # hm_s32_loss += self.crit(output['hm_s32'], batch['hm_s32']) / opt.num_stacks

            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_s_loss += (
                                       self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                    batch['dense_wh'] * batch['dense_wh_mask']) /
                                       mask_weight) / opt.num_stacks

                    mask_weight_s8 = batch['dense_wh_mask_s8'].sum() + 1e-4
                    wh_m_loss += (
                                          self.crit_wh(output['wh_s8'] * batch['dense_wh_mask_s8'],
                                                       batch['dense_wh_s8'] * batch['dense_wh_mask_s8']) /
                                          mask_weight_s8) / opt.num_stacks

                elif opt.cat_spec_wh:
                    wh_s_loss += self.crit_wh(
                        output['wh_s'], batch['cat_spec_mask'],
                        batch['ind_s'], batch['cat_spec_wh']) / opt.num_stacks

                    wh_m_loss += self.crit_wh(
                        output['wh_m'], batch['cat_spec_mask_s8'],
                        batch['ind_m'], batch['cat_spec_wh_s8']) / opt.num_stacks
                else:
                    wh_s_loss += self.crit_reg(
                        output['wh_s'], batch['reg_mask_s'],
                        batch['ind_s'], batch['wh_s']) / opt.num_stacks

                    wh_m_loss += self.crit_reg(
                        output['wh_m'], batch['reg_mask_m'],
                        batch['ind_m'], batch['wh_m']) / opt.num_stacks

                    wh_l_loss += self.crit_reg(
                        output['wh_l'], batch['reg_mask_l'],
                        batch['ind_l'], batch['wh_l']) / opt.num_stacks

                    # wh_s32_loss += self.crit_reg(
                    #   output['wh_s32'], batch['reg_mask_s32'],
                    #   batch['ind_s32'], batch['wh_s32']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_s_loss += self.crit_reg(output['reg_s'], batch['reg_mask_s'],
                                          batch['ind_s'], batch['reg_s']) / opt.num_stacks

                off_m_loss += self.crit_reg(output['reg_m'], batch['reg_mask_m'],
                                             batch['ind_m'], batch['reg_m']) / opt.num_stacks

                off_l_loss += self.crit_reg(output['reg_l'], batch['reg_mask_l'],
                                              batch['ind_l'], batch['reg_l']) / opt.num_stacks

                # off_s32_loss += self.crit_reg(output['reg_s32'], batch['reg_mask_s32'],
                #                               batch['ind_s32'], batch['reg_s32']) / opt.num_stacks

        loss = opt.hm_weight * hm_s_loss + opt.wh_weight * wh_s_loss + \
               opt.off_weight * off_s_loss + opt.hm_weight * hm_m_loss + 2 * opt.wh_weight * wh_m_loss + \
               opt.off_weight * off_m_loss + \
               opt.hm_weight * hm_l_loss + 4 * opt.wh_weight * wh_l_loss + \
               opt.off_weight * off_l_loss
        # opt.hm_weight * hm_s32_loss + opt.wh_weight * wh_s32_loss + \
        # opt.off_weight * off_s32_loss
        loss_stats = {'loss': loss, 'hm_s_loss': hm_s_loss,
                      'wh_s_loss': wh_s_loss, 'off_s_loss': off_s_loss,
                      'hm_m_loss': hm_m_loss,
                      'wh_m_loss': wh_m_loss, 'off_m_loss': off_m_loss,
                      'hm_l_loss': hm_l_loss,
                      'wh_l_loss': wh_l_loss, 'off_l_loss': off_l_loss,
                      # 'hm_s32_loss': hm_s32_loss,
                      # 'wh_s32_loss': wh_s32_loss, 'off_s32_loss': off_s32_loss
                      }
        return loss, loss_stats


class CtdetTrainer_ms(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer_ms, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_s_loss', 'wh_s_loss', 'off_s_loss', 'hm_m_loss', 'wh_m_loss', 'off_m_loss',
                       'hm_l_loss', 'wh_l_loss', 'off_l_loss', ]
        # 'hm_s32_loss', 'wh_s32_loss', 'off_s32_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]