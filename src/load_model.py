import torch

models = torch.load('/media/leo/data/code/pangnet/detection/object_as_point/Git/CenterNet/models/pangnet_branch.pth.tar')

vgg = torch.load('/home/leo/.torch/models/vgg16-397923af.pth')

weight = models['state_dict']

torch.save(weight, open('/media/leo/data/code/pangnet/detection/object_as_point/Git/CenterNet/models/pangnet_branch.pth', 'wb'))
