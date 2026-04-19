import torch
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision.transforms as T 


local_weights_path = "external_weights/dinov2_vits14_pretrain.pth" 

model = torch.hub.load(
    'facebookresearch/dinov2', 
    'dinov2_vits14', 
    pretrained=False
)

model.load_state_dict(torch.load(local_weights_path))
model.eval()



transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
])

yo =  Image.open('test.png').convert('RGB')
yo = transform(yo).unsqueeze(0)

with torch.no_grad():

    last_4_blokcks = model.get_intermediate_layers(yo,n=4)

