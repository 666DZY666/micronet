import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torchvision
import math
import time

# tensorrt-lib
import util_trt

# tensorrt eval/test
class SegmentationModule_v2_trt(nn.Module):
    def __init__(self, context, buffers, crit, deep_sup_scale=None, use_softmax=False, binding_id=0):
        super(SegmentationModule_v2_trt, self).__init__()
        self.context = context
        self.inputs, self.outputs, self.bindings, self.stream = buffers
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.use_softmax = use_softmax
        self.binding_id = binding_id
        
    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    # eval/test
    def forward(self, feed_dict, *, segSize=None, shape_of_input):
        shape_of_output = (1, 2, int(shape_of_input[2] / 8), int(shape_of_input[3] / 8))
        self.inputs[0].host = util_trt.to_numpy(feed_dict['img_data']).astype(dtype=np.float32).reshape(-1)
        trt_outputs, infer_time = util_trt.do_inference_v2(context=self.context, bindings=self.bindings, inputs=self.inputs, 
                outputs=self.outputs, stream=self.stream, h_=shape_of_input[2], w_=shape_of_input[3], binding_id=self.binding_id)
        trt_outputs = trt_outputs[0][:shape_of_output[0] * shape_of_output[1] * shape_of_output[2] * shape_of_output[3]]
        results = util_trt.postprocess_the_outputs(trt_outputs, shape_of_output)
        x = torch.from_numpy(results)
        x = x.cuda()
        if self.use_softmax: 
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        pred = x
        return pred, infer_time

class C1_unet_v3(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1_unet_v3, self).__init__()
        self.use_softmax = use_softmax

        #self.cbr1 = conv3x3_bn_relu(fc_dim//8, fc_dim // 4, 2)
        self.cbr1 = nn.Sequential(conv3x3_bn_relu(fc_dim, fc_dim, 1),
            conv3x3_bn_relu(fc_dim, fc_dim, 1),
            conv3x3_bn_relu(fc_dim, fc_dim, 1),
            )

        self.cbr2 = nn.Sequential(conv3x3_bn_relu(fc_dim//2, fc_dim // 2, 1),
                                  conv3x3_bn_relu(fc_dim // 2, fc_dim // 2, 1),
                                  conv3x3_bn_relu(fc_dim//2, fc_dim // 2, 1))

        self.cbr3 = conv3x3_bn_relu(fc_dim//2*3, fc_dim//2*3, 1)

        self.cbr4 = nn.Sequential(conv3x3_bn_relu(fc_dim//4, fc_dim // 4, 1),
                                  conv3x3_bn_relu(fc_dim // 4, fc_dim // 4, 1),
                                  conv3x3_bn_relu(fc_dim // 4, fc_dim // 4, 1))

        self.cbr5 = nn.Sequential(conv3x3_bn_relu(fc_dim//4*7, fc_dim //2, 1),
                                  conv3x3_bn_relu(fc_dim//2, fc_dim //2, 1))
        #self.conv_second = nn.Conv2d(fc_dim // 4, fc_dim // 4, 3, 1, 0)
        self.conv_last = nn.Conv2d(fc_dim // 2, num_class, 1, 1, 0)

        #self.upsample = Upsample(scale=2, mode='bilinear', align_corners=False)

    def forward(self, conv_out, segSize=None):
        shape = list(conv_out[1].size())
        shape =shape[2:]
        shape2 = list(conv_out[2].size())
        shape2 = shape2[2:]
        #x1 = self.cbr1(conv_out[0])
        x2 = self.cbr1(conv_out[3])
        #x2 = nn.functional.interpolate(
        #    x2, size=shape2, mode='bilinear', align_corners=False)
        x2 = nn.functional.interpolate(
            x2, size=(int(shape2[0]), int(shape2[1])), mode='bilinear', align_corners=False) # +++
        x3 = self.cbr2(conv_out[2])
        x3 = torch.cat([x2,x3],1)

        #x3 = nn.functional.interpolate(
        #    x3, size=shape, mode='bilinear', align_corners=False)
        x3 = nn.functional.interpolate(
            x3, size=(int(shape[0]), int(shape[1])), mode='bilinear', align_corners=False) # +++
        x3 = self.cbr3(x3)
        x4 = self.cbr4(conv_out[1])
        x4 = torch.cat([x4,x3],1)

        x5 = self.cbr5(x4)
        x = self.conv_last(x5)
        '''
        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        '''
        return x
        