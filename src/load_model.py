import torch

# models = torch.load('models/RESNET18BU_IMAGENET_CBAM_checkpoint.pth.tar')
models = torch.load('models/RESNET18BU_IMAGENET_CBAM_model_best.pth.tar')

vgg = torch.load('/home/leo/.torch/models/vgg16-397923af.pth')

weight = models['state_dict']

weight_rename = dict()

for w in weight:
    name_list = w.split('.')

    for i in range(len(name_list)):
        if i > 0:
            if i == 1:
                name = name_list[i] + '.'
            elif i > 1 and i < len(name_list) - 1:
                name += name_list[i] + '.'
            else:
                name += name_list[i]



    weight_rename[name] = weight[w]

    # weight_rename

torch.save(weight_rename, open('/media/leo/data/code/pangnet/detection/object_as_point/Git/CenterNet/models/resnet18_cbam_bu.pth', 'wb'))
