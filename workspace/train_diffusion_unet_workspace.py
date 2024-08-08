#@markdown ### **Training**
#@markdown
#@markdown Takes about 2.5 hours. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights
import numpy as np
import torch
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from model.netword import *
from model.diffussion_policy import Diffussion_policy
from model.vision_encoder import *
from dataset.dataset_my import *
from omegaconf import OmegaConf
import copy
import hydra
from torch.utils.data import DataLoader
import pathlib

OmegaConf.register_new_resolver("eval", eval, replace=True)

class Diffussion_Train:
    def __init__(self,
                 cfg: OmegaConf, output_dir=None
                 ) -> None:
        #参数
        self.cfg = cfg
        #权重路径
        if not (os.path.exists(self.cfg.checkpoint.path)):
            os.makedirs(self.cfg.checkpoint.path)
            
        #定义Diffusion模型
        self.model: Diffussion_policy = hydra.utils.instantiate(cfg.policy)
        #ema model
        self.ema_model: Diffussion_policy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        print("self.model:========================== ",self.model)
        #optimizer

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=self.model.model.parameters(),
            **optimizer_cfg
        )

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        #重启训练resunme

        #配置数据集
        dataset: PushTImageDataset
        dataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, PushTImageDataset) 
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))

        #学习率优化器
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_dataloader) * cfg.training.num_epochs
            )
        
        # configure ema配置EMA
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                parameters=self.ema_model.model.parameters())
            
        #train loop
        with tqdm(range(cfg.training.num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(train_dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in (tepoch):
                        loss = self.model.compute_loss(nbatch)

                        # optimize
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        ema.step(self.model.model.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                        
                if epoch_idx % cfg.checkpoint.save_epoch == 0:
                    torch.save(ema.state_dict(),os.path.join(self.cfg.checkpoint.path,f"model_{epoch_idx}_with_loss_{loss_cpu}.pth"))
                tglobal.set_postfix(loss=np.mean(epoch_loss))


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('config'))#去到./diffusion_policy/config/
)
def main(cfg: OmegaConf):#接收
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: Diffussion_Train = cls(cfg)#根据yaml中的__target__实例化
    workspace.run()

if __name__=="__main__":
    main()