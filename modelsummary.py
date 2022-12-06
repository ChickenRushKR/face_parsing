import torchsummary
from ibug.face_parsing.rtnet import *
import torch
# images = torch.rand(1, 3, 224, 224).cuda(0)
# rois=torch.rand(1, 4).cuda(0)
# model = rtnet50(images,rois)
# print(torchsummary.summary(model, (3,224,224)))
images = torch.rand(1, 3, 224, 224).cuda(0)
rois = torch.rand(1, 4).cuda(0)
model = rtnet50()
model = model.cuda(0)
print(model(images, rois).size())