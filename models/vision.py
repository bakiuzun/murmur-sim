import torch
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision.transforms as T 
import os 

LOCAL_WEIGHT_PATH = "external_weights/dinov2_vits14_pretrain.pth" 


class Vision():
    def __init__(self):


    self.model = torch.hub.load(
        'facebookresearch/dinov2', 
        'dinov2_vits14', 
        pretrained=False
    )
    self.model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH))
    self.model.eval()

    self.transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
    ])


    def get_features(self,img_folder):

        paths = os.listdir(img_folder)

        for p in paths:    
            img =  Image.open(f"{img_folder}/{p}").convert('RGB')
            img = transform(yo).unsqueeze(0)

            with torch.no_grad():

                last_4_blocks = model.get_intermediate_layers(img,n=4)
                print("Last 4 ",type(last_4_blocks))

vision = Vision()

vision.get_features("target_imgs")


