import wandb




def init_wandb(config,project="drone-rl",name=None):
    if name is None:
        name = f"lr{config['lr']}_envs{config['num_envs']}"

    wandb.init(project=project,config=config,name=name)


def log_metrics(metrics,step=None):
    wandb.log({k: float(v) for k,v in metrics.items()},step=step)


def finish_wandb():wandb.finish()



