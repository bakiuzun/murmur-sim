import torch
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision.transforms as T 
import os 
import torch.nn.functional as F
import numpy as np 
import os 


LOCAL_WEIGHT_PATH = "models/external_weights/dinov2_vits14_pretrain.pth" 

IMGNET_MEAN = [0.485,0.456,0.406] 
IMGNET_STD = [0.229,0.224,0.225]


# DEFAULT SHAPE  (256,384)
class VisionModule():
    def __init__(self,img_size=224):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = torch.hub.load(
            'facebookresearch/dinov2', 
            'dinov2_vits14', 
            pretrained=False
        )
        self.model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH))
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model = torch.compile(self.model,mode="reduce-overhead")


        self.transform = T.Compose([
            T.Resize((img_size,img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMGNET_MEAN,
                        std=IMGNET_STD),
        ])

        self.img_size = img_size

        self.mean = torch.tensor(IMGNET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMGNET_STD, device=self.device).view(1, 3, 1, 1)


        self.features_bank = None

    def _prepare_input(self, img):
        """
        Convertit n'importe quelle entrée en tensor (B, 3, H, W) normalisé ImageNet.
        Accepte : PIL Image, tensor (H,W,C) uint8, tensor (C,H,W) float, tensor (B,C,H,W) float.
        """
        # Cas 1 : PIL Image
        if isinstance(img, Image.Image):
            img = self.transform(img).unsqueeze(0)  # (1, 3, H, W) déjà normalisé
            return img.to(self.device)
        
        # Cas 2 : numpy array
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        
        img = img.to(self.device).float()
        
        # Si valeurs entre 0-255, ramener à 0-1
        if img.max() > 1.5:
            img = img / 255.0
        
        # Si (H, W, C), permute en (C, H, W)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        
        # Si (C, H, W), ajouter batch dim
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        # (B, H, W, C) -> (B, C, H, W) au cas où
        if img.ndim == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        
        # Resize si pas la bonne taille
        if img.shape[-1] != self.img_size or img.shape[-2] != self.img_size:
            img = F.interpolate(img, size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=False)
        
        # Normalisation ImageNet
        img = (img - self.mean) / self.std


        img = img.to(self.device)
        
        return img

    def load_features(self,feature_path):

        # it's not neural net weights.. just a tensor pre computed 
        self.features_bank = torch.load(feature_path,weights_only=False)  
        
        self.features_bank = self.features_bank.to(self.device)

        self.features_bank = F.normalize(self.features_bank,
                                         p=2.0,dim=-1)
        
        print(f"Loaded: {self.features_bank.shape[0]} features of dim: {self.features_bank.shape[1:]} ")

    def cosine_sim(self,input_x,compute_features=True):
        """
        cosine sim with the bank features.
        We hope that more the cosine sim is high and more the drone look
        at the target. this will be our reward basically
        """
        if self.features_bank is None:
            raise Exception('Features bank should be computed before calling cosine sim')

        
        img_features = input_x
        n_envs = input_x.shape[0]

        if compute_features:
            img_features = self.get_features(input_x)
            
        # we might need to normalize for cos sim 
        normalized_img_feat = F.normalize(img_features,p=2.0,dim=-1)
        
        D = normalized_img_feat.shape[-1]

        normalized_img_feat = normalized_img_feat.reshape(-1,D)
        self.features_bank = self.features_bank.reshape(-1,D)
        
        # (10*256,386) @ (386,44*256) -> (256,44*256)    
        cos_sim = normalized_img_feat @ self.features_bank.T 

        # 10*256,100000 -> 10*256
        best_per_patch = cos_sim.max(dim=1).values

        # 10*256 -> 1 (scalar)
        #the_most_similar = best_per_patch.max()
        mean_cos_per_env = best_per_patch.reshape(n_envs,-1).mean(dim=1)
        
        return mean_cos_per_env
    
    def get_features(self,img):

        img = self._prepare_input(img)

        """
        with torch.no_grad():
            # blocks (1,256,384) 
            all_blocks = self.model.get_intermediate_layers(img,n=12)
        """
        with torch.no_grad(),torch.autocast(device_type=self.device,dtype=torch.float16):
            cls_token = self.model(img)

        return cls_token.float()

        
    def save_features(self,img_folder:str,save_path:str):
        paths = os.listdir(img_folder)
        os.makedirs(save_path,exist_ok=True)

        all_features = []


        for p in paths:    
            img = Image.open(f"{img_folder}/{p}").convert('RGB')

            features = self.get_features(img)

            all_features.append(features)
            
        all_features = torch.cat(all_features,dim=0)

        torch.save(all_features,f"{save_path}/features.pt")

        print("All features has been saved!")


    def e_greedy(self,save_path,ratio=0.1):
        if self.features_bank is None:
            print("Skipping e greedy you must first load features bank")
            return 
        N = self.features_bank.shape[0]
        total_samples = int(N * ratio)

        random_index = torch.randint(0,N,size=(1,))

    
        current = self.features_bank[random_index] 
        samples = [current]

        minimum = torch.full(size=(N,1),fill_value=float('inf'))
        minimum[random_index] = 0.
        for _ in range(total_samples - 1):

            # (1,59)
            dist = torch.cdist(self.features_bank,current)
            

            minimum = torch.minimum(dist,minimum)

            maxx_dist_index = torch.argmax(minimum,dim=0)

            minimum[maxx_dist_index] = 0. 

            current = self.features_bank[maxx_dist_index]
            samples.append(current)
        
        samples = torch.cat(samples,dim=0)
        
        torch.save(samples,save_path)
        
        print("Saved e greedy samples to: ",save_path)

        return samples 


if __name__ == "__main__":
    vision = VisionModule()

    vision.save_features('target_imgs/',save_path='target_features')
    vision.load_features('target_features/features.pt')
    vision.e_greedy(save_path='target_features/e_greedy_features.pt',ratio=0.1)

