import gymnasium as gym 


class UAVEnvironement(gym.Env):
    metada = {'render_modes':['none','human']}
    def __init__(self,render_mode='none'):
        super().__init__()

        self.render_mode = render_mode 
        
        ## action space 
        ## observation space 

        # QUEL ACTIONS DOIS TIL RETOURNER ET CES QUOI MES OBS 
        # TAKE OFF, il dois se lever et se stabiliser 

        # jenvoie des commands le FC écoute sa et transforme en 
        # truc electrique que le ESC va envoyer 
        # quesque le FC ressoie exactement ? 1 
        # de 2 quelles sont les possitilitées ? 
        # je veut pas mettre de contraintes dure reste à 1.5M 
        # faudrait tout de même qu'il se lève et prend une altitude
        # et puis il reste sur cette altitude 
        # je peux directement essayer de modifier genre en donnant
        # des contrôles. 
        # Ok on va tout d'abord V1 essayer de le stabiliser a 2M de hauteur 
        
    def step(self,action): 
        """
        obs,reward,terminated,truncated,info
        """


        return self.step(action)


    def reset(self,seed=None):
        super().reset(seed=seed)

        """

        return observation and info 
        """
    


    def render(self):pass


    def close(self):pass
