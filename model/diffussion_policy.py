import torch
import torch.nn as nn
from model.netword import *
from model.vision_encoder import *
from dataset.dataset_my import *


class Diffussion_policy:
    def __init__(self,
                noise_scheduler,
                vision_encoder_name: str = "resnet18",
                vision_feature_dim: int = 512,
                lowdim_obs_dim :int = 2,
                action_dim: int = 2,
                obs_horizon: int = 2,
                device: str = "cuda:0",
                
                 ) -> None:
        
        self.vision_encoder_name = vision_encoder_name
        self.vision_feature_dim = vision_feature_dim
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim  = action_dim 
        self.obs_dim = vision_feature_dim + lowdim_obs_dim
        self.obs_horizon = obs_horizon

        self.device = device

        vision_encoder = get_resnet(self.vision_encoder_name)
        vision_encoder = replace_bn_with_gn(vision_encoder)

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        )
        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        })

        self.noise_scheduler = noise_scheduler
        self.model = nets.to(self.device)

    def compute_loss(self, nbatch):
        nimage = nbatch['image'][:,:self.obs_horizon].to(self.device)
        nagent_pos = nbatch['agent_pos'][:,:self.obs_horizon].to(self.device)
        naction = nbatch['action'].to(self.device)
        B = nagent_pos.shape[0]

        # encoder vision features
        image_features = self.model['vision_encoder'](
            nimage.flatten(end_dim=1))
        image_features = image_features.reshape(
            *nimage.shape[:2],-1)
        # (B,obs_horizon,D)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            naction, noise, timesteps)

        # predict the noise residual
        noise_pred = self.model["noise_pred_net"](
            noisy_actions, timesteps, global_cond=obs_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss