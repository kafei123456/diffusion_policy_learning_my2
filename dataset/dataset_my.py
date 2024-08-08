#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTImageDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a zarr storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)
import numpy as np
import torch
import zarr
import os
import gdown
import cv2
def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    """
    params:
        episode_ends: 图像
        sequence_length： 序列长度
        pad_before： padding 前
        pad_after： paddling 后
    
    """
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]   
        end_idx = episode_ends[i]   #161
        episode_length = end_idx - start_idx#161 - 0 = 161

        min_start = -pad_before #-1
        max_start = episode_length - sequence_length + pad_after    #161-16+(8-1) = 161-9

        # range stops one idx before end 范围在结束前停止一个idx
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    """
    params:
        train_data: self.normalized_train_data包含image，agent_pos，action
        sequence_length: self.pred_horizon,预测的序列长度
        buffer_start_idx: buffer_start_idx,开始的idx
        buffer_end_idx: buffer_end_idx,结束的idx
        sample_start_idx: sample_start_idx,采用的开始idx
        sample_end_idx: sample_end_idx，采用的结束idx
    out:
        result:
    """
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0] #开始的时候，初始的状态值多复制一个
            if sample_end_idx < sequence_length:#最后几个数据
                data[sample_end_idx:] = sample[-1]#补充为最后一个值
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    #print()
    return result

# normalize data数据归一化
def get_data_stats(data):
    """
    descript:获取数据的最大最小值

    """
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    """
    descript:把数据归一化到[-1,1]
    """
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    """
    descript:数据反归一化
    """
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # float32, [0,1], (N,96,96,3)，具有N张图像
        train_image_data = dataset_root['data']['img'][:]
        #print("train_image_data[20000]：",train_image_data.shape)
        # print("train_image_data[20000]：",train_image_data[20000][0][0])
        # cv2.imwrite("d.png",cv2.resize(train_image_data[20000], (512,512)))
        #cv2.waitKey(0)
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96)
        # print('agent_pos',dataset_root['data']['state'][0:100,:2])
        # print('action',dataset_root['data']['action'][0:100])
        # (N, D)N个数据
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations状态向量的前两个dims是agent（即夹持器）位置
            'agent_pos': dataset_root['data']['state'][:,:2],
            'action': dataset_root['data']['action'][:]
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]
        #print("episode_ends: ",episode_ends)
        # compute start and end of each state-action sequence# 计算每个状态-动作序列的开始和结束
        # also handles padding# 还处理填充
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)
        # print("indices: \n",indices)
        # print("indices: \n",indices[0:200])
        # print("indices: \n",indices[-100:])
        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])#归一化agent_pos和action

        # images are already normalized归一化图像（图像在数据集中以及归一化了）
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data#包含了(agent_pos\action\image)
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        # print("self.indices: ",self.indices)
        #print(f"buffer_start_{idx} = ",buffer_start_idx, "——",buffer_end_idx)
        #print(f"sample_start_{idx} = ",sample_start_idx, "——",sample_end_idx)
        # get nomralized data using these indices使用这些indices获取归一化的数据
        nsample = sample_sequence(#采样序列
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        #print("agent_pos: ",nsample['agent_pos'])
        #print("action: ", nsample['action'])
        # discard unused observations丢弃未使用的观测值
        nsample['image'] = nsample['image'][:self.obs_horizon,:]#让image的horizon为2
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]#让agent_pos的horizon为2
        return nsample
    
if __name__=="__main__":
    #@markdown ### **Dataset Demo**

    # download demonstration data from Google Drive
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("agent_pos: ",batch['agent_pos'])
    print("action: ",batch['action'])
    #print("batch['image'].shape:", batch['image'])
    print("batch['image'].shape:", batch['image'].shape)
    print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
    print("batch['action'].shape", batch['action'].shape)

    # for i in range(5):
    #     batch = next(iter(dataloader))
    #     print("agent_pos: ",batch['agent_pos'])
    #     print("action: ",batch['action'])