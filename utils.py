import pickle 
from flax import nnx 
import wandb 


def save_model(filepath, params: nnx.statelib.State, non_params:nnx.statelib.State):
    with open(filepath, 'wb') as f:
        pickle.dump({"params": params, "non_params": non_params}, f)

def load_model(filepath, graphdef:nnx.graph.GraphDef):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return nnx.merge(graphdef, data["params"], data["non_params"])



def init_wandb(config,project="drone-rl",name=None):
    if name is None:
        name = f"lr{config['lr']}_envs{config['num_envs']}"

    wandb.init(project=project,config=config,name=name)


def log_metrics(metrics,step=None):
    wandb.log({k: float(v) for k,v in metrics.items()},step=step)


def finish_wandb():wandb.finish()



