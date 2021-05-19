import numpy as np
from torch.nn import functional as F
from torch import nn
from torchvision.ops import nms
import torch


class Region_Proposal_Network(nn.Module):
    def __init__(self, in_channels, out_channels, subsampling, scales, ratios, stride):
        super(Region_Proposal_Network, self).__init__()
        # input and output channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        # anchor scales and ratios, expected a list like [8,16,32] and [0.5,1,2] respectively 
        self.scales = scales
        self.ratios = ratios
        # k anchor boxes
        self.k = len(scales) * len(ratios)
        # subsampling factor, VGG16: 16, 800->50
        self.subsampling = subsampling
        # stride
        self.stride = stride
        # generate k anchor boxes on the feature map (subsampled)
        # EX: [22500, 4]
        self.anchor_box = self.get_anchor_box()
        # intermediate layer
        self.intermediate = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # reg layer outputs 4k, 4 box coordinates of k boxes
        # deault: stride=1, padding=0
        self.reg = nn.Conv2d(out_channels, 4*self.k, kernel_size=1)
        # cls layer outputs 2k scores that estimate probability of object or not object
        # deault: stride=1, padding=0
        self.cls = nn.Conv2d(out_channels, 2*self.k, kernel_size=1)
        # initialize all new layers by drawing weights from a zero-mean Gaussian distribution with standard deviation 0.01
        self.intermediate.weight.data.normal_(0, 0.01)
        self.intermediate.bias.data.zero_()
        self.reg.weight.data.normal_(0, 0.01)
        self.reg.bias.data.zero_()
        self.cls.weight.data.normal_(0, 0.01)
        self.cls.bias.data.zero_()
    
    def forward(self, x, image_size, scale=1.):
        # N: batch size, C: channel, H,W: height and width of input feature map
        N, C, H, W = x.shape
        # generate all anchor boxes on the original image space
        anchor = self.get_all_anchor_box(H,W)
        x = F.relu(self.intermediate(x))
        pred_anchor_locs  = self.reg(x) # [1,4*9,50,50]
        pred_cls_scores = self.cls(x) # [1,2*9,50,50]
        # reshape to fit anchor targets, [1,36,50,50] -> [1,50,50,36] -> [1,22500,4]
        pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        # reshape to fit anchor targets, [1,18,50,50] -> [1,50,50,18]
        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
        # dim=4: normalize output score
        # cls layer as a two-class softmax layer
        pred_softmax_scores = F.softmax(pred_cls_scores.view(N, H, W, self.k, 2), dim=4)
        # EX: [1, 22500]
        objectness_score = pred_softmax_scores[:, :, :, :, 1].contiguous().view(N, -1)
      
        # Proposal Layer 
        # generate a set of proposals
        # loop through batch, feed one image at a time
        pred_boxes = []
        pred_boxes_idx = []
        for i in range(N):
            pred_box = self.generate_proposals(pred_anchor_locs[i].cpu().data.numpy(), anchor,objectness_score[i].cpu().data.numpy(), image_size)
            batch_idx = i * np.ones((len(pred_box), ), dtype=np.int32)
            pred_boxes.append(pred_box)
            pred_boxes_idx.append(batch_idx)

        pred_boxes = np.concatenate(pred_boxes, axis=0)
        pred_boxes_idx = np.concatenate(pred_boxes_idx, axis=0)

        # pred_cls_scores: [1,50,50,18] -> [1,22500,2]
        return pred_anchor_locs, pred_cls_scores.view(N, -1, 2), pred_boxes, pred_boxes_idx, anchor

    # 针对原image 800x800 全部生成的anchor box的坐标点
    # EX: [ -37.254833  -82.50967    53.254833   98.50967 ]
    # [22500,4] = 800x800/(16x16)=2500,2500*9=22500
    def get_all_anchor_box(self, H, W):
        shift_x = np.arange(0, W*self.stride, self.stride)
        shift_y = np.arange(0, H*self.stride, self.stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # shift.shape (2500,4)
        shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                          shift_y.ravel(), shift_x.ravel()), axis=1)
  
        A = self.anchor_box.shape[0]
        K = shift.shape[0]
        anchor_box = self.anchor_box
        anchor = anchor_box.reshape((1, A, 4)) + \
                 shift.reshape((1, K, 4)).transpose((1, 0, 2))
        anchor = anchor.reshape((K * A, 4)).astype(np.float32)

        return anchor
  
    # 针对原image 16X16 region生成的9个anchor box  
    # EX: [ -37.254833  -82.50967    53.254833   98.50967 ]
    def get_anchor_box(self): 
        anchor_box = np.zeros((len(self.scales)*len(self.ratios), 4), dtype=np.float32)
        x_center  = self.subsampling / 2.
        y_center  = self.subsampling / 2.
        for i in range(len(self.ratios)):
            for j in range(len(self.scales)):
                h = self.subsampling * self.scales[j] * np.sqrt(self.ratios[i])
                w = self.subsampling * self.scales[j] * np.sqrt(1./ self.ratios[i])
                index = i * len(self.scales) + j
                # 0,1 -> top left coordinates, 2,3->bottom right coordinates
                # [y1,x1,y2,x2]
                anchor_box[index, 0] = y_center - h / 2.
                anchor_box[index, 1] = x_center - w / 2.
                anchor_box[index, 2] = y_center + h / 2.
                anchor_box[index, 3] = x_center + w / 2.
    
        return anchor_box
  
  
    def generate_proposals(self, pred_anchor_locs, anchor, objectness_score, image_size):
        # IoU threshold for NMS
        nms_thresh = 0.7
        if(self.training):  # During training
            num_pre_nms = 12000 # number of anchor box before nms
            num_after_nms = 2000 # number of anchor box after nms
        else:               # During testing
            num_pre_nms = 6000 # number of anchor box before nms
            num_after_nms = 300 # number of anchor box after nms
            
        # minimum height of the object required to create a proposal
        min_size = 16
        # convert pred_anchor_locs to anchor_box coordinates [y1, x1, y2, x2] format
        # [1,22500,4] -> 
            # compute center of the anchor box on 800x800 image space
        # and height and width of the anchor box
        anchor = anchor.astype(anchor.dtype, copy=False)
        h = anchor[:, 2] - anchor[:, 0]
        w = anchor[:, 3] - anchor[:, 1]
        y_center = anchor[:, 0] + 0.5 * h
        x_center = anchor[:, 1] + 0.5 * w
        
        
        # outputs of the reg layer
        # [:, 0::4] to preserve 2D array
        dy = pred_anchor_locs[:, 0::4]
        dx = pred_anchor_locs[:, 1::4]
        dh = pred_anchor_locs[:, 2::4]
        dw = pred_anchor_locs[:, 3::4]
        # compute new center of the predicted anchor box by reg layer
        new_y_center = dy * h[:, np.newaxis] + y_center[:, np.newaxis]
        new_x_center = dx * w[:, np.newaxis] + x_center[:, np.newaxis]
        new_h = np.exp(dh) * h[:, np.newaxis]
        new_w = np.exp(dw) * w[:, np.newaxis]
        pred_box = np.zeros(pred_anchor_locs.shape, dtype=pred_anchor_locs.dtype)
        pred_box[:, 0::4] = new_y_center - 0.5 * new_h
        pred_box[:, 1::4] = new_x_center - 0.5 * new_w
        pred_box[:, 2::4] = new_y_center + 0.5 * new_h
        pred_box[:, 3::4] = new_x_center + 0.5 * new_w
        # make sure the predicted anchor box stay inside the image
        pred_box[:, 0] = np.clip(pred_box[:, 0], 0, image_size[0]) # [0,H]
        pred_box[:, 2] = np.clip(pred_box[:, 2], 0, image_size[0]) 
        pred_box[:, 1] = np.clip(pred_box[:, 1], 0, image_size[1]) # [0,W]
        pred_box[:, 3] = np.clip(pred_box[:, 3], 0, image_size[1])
        # Remove predicted boxes with either height or width less than minimum size
        height_diff = pred_box[:, 2] - pred_box[:, 0]
        width_diff = pred_box[:, 3] - pred_box[:, 1]
        idx = np.where((height_diff >= min_size) & (width_diff >= min_size))[0]
        pred_box = pred_box[idx, :]
        objectness_score = objectness_score[idx] # [22500, 1]
        # select num_pre_nms number of anchor box
        # EX: 22500 -> 12000
        idx = np.argsort(objectness_score.ravel())[::-1]
        idx = idx[0:num_pre_nms]
        pred_box = pred_box[idx, :]
        objectness_score = objectness_score[idx]

        # perform Non-maximal suppression
        # Cuda or CPU? 
        # TODO: move to cpu?
        nms_result = nms(torch.from_numpy(pred_box).cuda(), torch.from_numpy(objectness_score).cuda(), nms_thresh)
        nms_result[:num_after_nms]
        pred_box = pred_box[nms_result.cpu().numpy()]
        return pred_box