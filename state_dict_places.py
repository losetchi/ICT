import torch
from collections import OrderedDict
state_dict = torch.load("/content/ICT/ckpts_ICT/Upsample/ImageNet/InpaintingModel_gen.pth", map_location='cpu')
new_state_dict = OrderedDict()

for k, v in state_dict['generator'].items():
  name = k.replace("module.", "")

  new_state_dict[name] = v

torch.save(new_state_dict, '/content/ICT/ckpts_ICT/Upsample/ImageNet/InpaintingModel_gen.pth')

state_dict = torch.load("/content/ICT/ckpts_ICT/Upsample/Places2_Nature/InpaintingModel_gen.pth", map_location='cpu')
new_state_dict = OrderedDict()

for k, v in state_dict['generator'].items():
  name = k.replace("module.", "")

  new_state_dict[name] = v

torch.save(new_state_dict, '/content/ICT/ckpts_ICT/Upsample/Places2_Nature/InpaintingModel_gen.pth')