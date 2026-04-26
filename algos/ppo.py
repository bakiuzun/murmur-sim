import torch 
from envs import UAVEnv
from models import ActorCritic 
import utils 
from structs import Transition,ModelSpec,EnvState
from eval import metrics
from torch import optim
import torch.nn as nn 


class PPO():
    def __init__(self,config,actorSpec=None,criticSpec=None):

        self.config = config 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_envs = self.config['num_envs']

        self._prepare_train(config,actorSpec,criticSpec)
        print("Training has been prepared")

        self.obs = self.env.reset()

    def _prepare_train(self,config,actorSpec=None,criticSpec=None,ckpt_path=None):
        self.env = UAVEnv(config) 

        if actorSpec is None:
            actorSpec = ModelSpec(
                hidden_sizes=[64, 64, self.env.act_size],
                hidden_activation=nn.ReLU(),
                last_activation=config['actor_last_activation']
            )

        if criticSpec is None: 
            criticSpec = ModelSpec(
                hidden_sizes=[64, 64, 1],
                hidden_activation=nn.ReLU(),
                last_activation=None
            )
        self.model = ActorCritic(obs_dim=self.env.obs_size,
                        action_dim=self.env.act_size,
                        actor_spec=actorSpec,
                        critic_spec=criticSpec)

        
        self.num_steps = config['num_steps']
        self.num_updates = int(config['total_timesteps'] // self.num_steps // config['num_envs'])
        self.total_optimizer_steps = self.num_updates * config['update_epochs'] * config['num_minibatches']

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'])

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_optimizer_steps,
            eta_min=config['lr'] * 0.1
        )


    @torch.no_grad()
    def rollout(self):
        obs_buffer = torch.zeros((self.num_steps, self.num_envs, self.env.obs_size)).to(self.device)
        actions_buffer = torch.zeros((self.num_steps, self.num_envs, self.env.act_size)).to(self.device)
        logprobs_buffer = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards_buffer = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values_buffer = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones_buffer = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        for i in range(self.num_steps):
            action, value, log_prob = self.model(self.obs, action=None)
                
            obs_buffer[i] = self.obs
            actions_buffer[i] = action
            logprobs_buffer[i] = log_prob
            values_buffer[i] = value.flatten() 

            new_obs, reward, terminated, truncated = self.env.step(action)
            
            done = terminated | truncated # Using our fix from earlier!

            rewards_buffer[i] = reward
            dones_buffer[i] = terminated

            if done.any():
                done_indices = done.nonzero(as_tuple=True)[0]
                new_obs[done_indices] = self.env.reset(done_indices)

            
            self.obs = new_obs 
  
        return obs_buffer, actions_buffer, logprobs_buffer, rewards_buffer, values_buffer, dones_buffer




    @torch.no_grad()
    def calculate_gae(self,rewards_b, values_b, dones_b, next_obs, model, config):
        """
        Calculates Generalized Advantage Estimation (GAE).
        Assumes buffers are shape (num_steps, num_envs).
        """
        # 1. Bootstrap the final value
        # Get the value of the state AFTER the final step in the rollout
        _, next_value, _ = model(next_obs, action=None)
        
        next_value = next_value.squeeze(-1) 
        
        advantages = torch.zeros_like(rewards_b)
        lastgaelam = 0
        num_steps = len(rewards_b)
        
        for t in reversed(range(num_steps)):
            
            if t == num_steps - 1:
                nextvalues = next_value
            else:
                nextvalues = values_b[t + 1]
                
            mask = 1.0 - dones_b[t]
            
            # Core GAE Math
            delta = rewards_b[t] + config['gamma'] * nextvalues * mask - values_b[t]
            lastgaelam = delta + config['gamma'] * config['gae_lambda'] * mask * lastgaelam
            
            advantages[t] = lastgaelam
            
        # 4. Calculate Returns (Targets for the Value Network)
        returns = advantages + values_b
        
        # 5. Normalize Advantages (Crucial for PPO stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns


    def update(self, obs_b, actions_b, logprobs_b, advantages_b, returns_b):
        clip_eps = self.config['clip_eps']
        

        batch_size = self.config['num_steps'] * self.config['num_envs']
        
        b_obs = obs_b.reshape((-1, self.env.obs_size))
        b_actions = actions_b.reshape((-1, self.env.act_size))
        b_logprobs = logprobs_b.reshape(-1)
        b_advantages = advantages_b.reshape(-1)
        b_returns = returns_b.reshape(-1)

        # 2. Calculate Minibatch sizes
        num_minibatches = self.config['num_minibatches']
        minibatch_size = batch_size // num_minibatches

        # 3. Outer loop: Iterate over PPO Epochs
        for epoch in range(self.config['update_epochs']):
            
            # Shuffle indices for this epoch
            b_inds = torch.randperm(batch_size, device=self.device)
            
            # 4. Inner loop: Iterate over Minibatches
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Extract the minibatch data
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]

                # --- FORWARD PASS ---
                _, new_value, new_log_prob = self.model(mb_obs, action=mb_actions)
                new_value = new_value.squeeze(-1)

                # --- POLICY LOSS ---
                ratio = torch.exp(new_log_prob - mb_logprobs)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- VALUE LOSS ---
                v_loss = ((new_value - mb_returns) ** 2).mean()

                # --- TOTAL LOSS ---
                loss = pg_loss + 0.5 * v_loss

                # --- BACKPROPAGATION ---
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              max_norm=self.config['max_grad_norm'])
                
                self.optimizer.step()
                self.scheduler.step()

    def train(self):
        
        for update_step in range(self.num_updates):

            obs_b, actions_b, logprobs_b, rewards_b, values_b, dones_b = self.rollout()

            advantage, returns = self.calculate_gae(rewards_b,
                                                    values_b,
                                                    dones_b,
                                                    self.obs, 
                                                    self.model,
                                                    self.config)

            
            self.update(obs_b, actions_b, logprobs_b, advantage, returns)
            
            print(f"Update {update_step}/{self.num_updates} complete.")
            if update_step % 100 == 0:
                torch.save(self.model.state_dict(),f'new_model_{update_step}.pt')


            results = metrics.compute_metrics(obs_b.reshape(-1,self.env.obs_size),
                                    actions_b.reshape(-1,self.env.act_size),
                                    rewards_b.reshape(-1),
                                    sucess_counter=self.env.success_counter)

            
            utils.log_metrics(results)
            

        torch.save(self.model.state_dict(),'new_model.pt')

