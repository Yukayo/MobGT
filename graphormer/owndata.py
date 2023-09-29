import os
import os.path as osp
import shutil
import pickle

import torch
import pandas as pd
import math
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from collections import deque, Counter
from itertools import repeat, product
import numpy as np

seed=1

def collate(data_list):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(
                    item.__cat_dim__(key, item[key]))
            else:
                s = slices[key][-1] + 1
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            item = data_list[0][key]
            if torch.is_tensor(item) and len(data_list) > 1:
                if key in ['time', 'time_normal', 'user', 'cat']:
                    data[key] = torch.cat(data[key],
                                      dim=0)
                else:
                    data[key] = torch.cat(data[key],
                                      dim=data.__cat_dim__(key, item))
            elif torch.is_tensor(item):  # Don't duplicate attributes...
                data[key] = data[key][0]
            elif isinstance(item, int) or isinstance(item, float):
                data[key] = torch.tensor(data[key])

            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices

def generate_queue(train_idx, mode,mode2):
        user = list(train_idx.keys()) #获取所有用户ID
        train_queue = deque()
        np.random.seed(seed)
        if mode == 'random':
            initial_queue = {}
            for u in user:
                if mode2 == 'train':
                    # initial_queue[u] = deque(train_idx[u][1:]) #[1:]
                    initial_queue[u] = deque(train_idx[u]) #[1:]
                else:
                    initial_queue[u] = deque(train_idx[u])
            queue_left = 1
            while queue_left > 0:
                np.random.shuffle(user) 
                for j, u in enumerate(user):
                    if len(initial_queue[u]) > 0:
                        train_queue.append((u, initial_queue[u].popleft()))
                    if j >= int(0.01 * len(user)): 
                        break
                queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0]) 
        elif mode == 'normal':
            for u in user:
                for i in train_idx[u]:
                    train_queue.append((u, i))
        return train_queue 

class MyData(Data):
     def __cat_dim__(self, key, value, *args, **kwargs):
         if key == 'time':
             return None
         else:
             return super().__cat_dim__(key, value, *args, **kwargs)


class Foursquare(InMemoryDataset):
    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        # assert split in ['train', 'val', 'test']
        assert split in ['train', 'test']
        super(Foursquare, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'test.pickle', 'train_idx.pkl', 'test_idx.pkl'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'whole'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        
        for split in ['train','test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))
            
            # if split=='train_all':
            #     split='train'
            # if split=='test_all':
            #     split='test'

            with open(osp.join(self.raw_dir, f'{split}_idx.pkl'), 'rb') as f:
                dict1 = pickle.load(f)
                if split=='train':
                    indices=generate_queue(dict1, 'random', split)
                else:
                    indices=generate_queue(dict1, 'normal', split)                                         

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            if split=="train":
                self.train_indices=indices
            # elif split=="val":
            #     self.val_indices=indices
            else:
                self.test_indices=indices

            data_list = []
            for idx in indices:
                # if len(mols[idx[0]])==idx[1] and split=="train":
                #     mol = mols[idx[0]][idx[1]] 
                # elif split=="train":
                #     mol = mols[idx[0]][idx[1]+1] 
                # else:
                    # mol = mols[idx[0]][idx[1]]
                mol = mols[idx[0]][idx[1]]
                x = mol['node_name'].to(torch.long).view(-1, 1)
                y = mol['target'].to(torch.long)

                adj = mol['edge_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
                
                # node_index=mol['node_index'].to(torch.long).view(-1,1)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                y=y)

                # data = MyData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                # y=y, time=mol['time'])

                data.time=mol['time'].to(torch.long).view(-1, 1) # 如果是max time的话要.to(torch.long).view(-1, 1)
                # data.time=mol['time']

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)
                
            pbar.close()


            torch.save(collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

            # torch.save(self.collate(data_list),
                    #    osp.join(self.processed_dir, f'{split}.pt'))
        print(0)

class Toyota(InMemoryDataset):
    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super(Toyota, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'test.pickle', 'train_idx.pkl', 'test_idx.pkl','val.pickle','val_idx.pkl'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'whole'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt','val.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        
        for split in ['train', 'val','test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))
            
            # if split=='train_all':
            #     split='train'
            # if split=='test_all':
            #     split='test'

            with open(osp.join(self.raw_dir, f'{split}_idx.pkl'), 'rb') as f:
                dict1 = pickle.load(f)
                if split=='train':
                    indices=generate_queue(dict1, 'random', split)
                else:
                    indices=generate_queue(dict1, 'normal', split)                                         

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            if split=="train":
                self.train_indices=indices
            elif split=="val":
                self.val_indices=indices
            else:
                self.test_indices=indices

            data_list = []
            for idx in indices:
                # if len(mols[idx[0]])==idx[1] and split=="train":
                #     mol = mols[idx[0]][idx[1]] 
                # elif split=="train":
                #     mol = mols[idx[0]][idx[1]+1] 
                # else:
                    # mol = mols[idx[0]][idx[1]]
                mol = mols[idx[0]][idx[1]]
                x = mol['node_name'].to(torch.long).view(-1, 1)
                y = mol['target'].to(torch.long)

                adj = mol['edge_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)
                
            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
        print(0)

class FoursquareGraph(InMemoryDataset):
    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        # assert split in ['train', 'val', 'test']
        assert split in ['train', 'test']
        super(FoursquareGraph, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'test.pickle', 'train_idx.pkl', 'test_idx.pkl'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'whole'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        
        for split in ['train','test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            with open(osp.join(self.raw_dir, f'{split}_idx.pkl'), 'rb') as f:
                dict1 = pickle.load(f)
                if split=='train':
                    indices=generate_queue(dict1, 'random', split)
                else:
                    indices=generate_queue(dict1, 'normal', split)                                         

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            if split=="train":
                self.train_indices=indices
            # elif split=="val":
            #     self.val_indices=indices
            else:
                self.test_indices=indices

            data_list = []
            for idx in indices:
                mol = mols[idx[0]][idx[1]]
                x = mol['node_name'].to(torch.long).view(-1, 1)
                y = mol['target'].to(torch.long)

                adj = mol['edge_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
                

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                y=y)

                data.time=mol['time'].to(torch.long).view(-1, 1) # 如果是max time的话要.to(torch.long).view(-1, 1)
                data.time_normal=mol['time_normal'].to(torch.float).view(-1, 1) 
                data.user=mol['user'].to(torch.long).view(-1, 1) 
                data.cat=mol['cat'].to(torch.long).view(-1, 1) 
                # data.time=mol['time']

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)
                
            pbar.close()
            torch.save(collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

        print(0)

class GowallaGraph(InMemoryDataset):
    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        # assert split in ['train', 'val', 'test']
        assert split in ['train', 'test']
        super(GowallaGraph, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'test.pickle', 'train_idx.pkl', 'test_idx.pkl'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'whole'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        
        for split in ['train','test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            with open(osp.join(self.raw_dir, f'{split}_idx.pkl'), 'rb') as f:
                dict1 = pickle.load(f)
                if split=='train':
                    indices=generate_queue(dict1, 'random', split)
                else:
                    indices=generate_queue(dict1, 'normal', split)                                         

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            if split=="train":
                self.train_indices=indices
            # elif split=="val":
            #     self.val_indices=indices
            else:
                self.test_indices=indices

            data_list = []
            for idx in indices:
                mol = mols[idx[0]][idx[1]]
                x = mol['node_name'].to(torch.long).view(-1, 1)
                y = mol['target'].to(torch.long)

                adj = mol['edge_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
                

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                y=y)

                data.time=mol['time'].to(torch.long).view(-1, 1) # 如果是max time的话要.to(torch.long).view(-1, 1)
                data.time_normal=mol['time_normal'].to(torch.float).view(-1, 1) 
                data.user=mol['user'].to(torch.long).view(-1, 1) 
                data.cat=mol['cat'].to(torch.long).view(-1, 1) 
                # data.time=mol['time']

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)
                
            pbar.close()
            torch.save(collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

        print(0)


# ADDED ToyotaGraph from graphormer_new_cuda2
class ToyotaGraph(InMemoryDataset):
    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        # assert split in ['train', 'val', 'test']
        assert split in ['train', 'test']
        super(ToyotaGraph, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'test.pickle', 'train_idx.pkl', 'test_idx.pkl'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'whole'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        
        for split in ['train','test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            with open(osp.join(self.raw_dir, f'{split}_idx.pkl'), 'rb') as f:
                dict1 = pickle.load(f)
                if split=='train':
                    indices=generate_queue(dict1, 'random', split)
                else:
                    indices=generate_queue(dict1, 'normal', split)                                         

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            if split=="train":
                self.train_indices=indices
            # elif split=="val":
            #     self.val_indices=indices
            else:
                self.test_indices=indices

            data_list = []
            for idx in indices:
                mol = mols[idx[0]][idx[1]]
                x = mol['node_name'].to(torch.long).view(-1, 1)
                y = mol['target'].to(torch.long)

                adj = mol['edge_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
                

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                y=y)

                data.time=mol['time'].to(torch.long).view(-1, 1) # 如果是max time的话要.to(torch.long).view(-1, 1)
                data.time_normal=mol['time_normal'].to(torch.float).view(-1, 1) 
                data.user=mol['user'].to(torch.long).view(-1, 1) 
                data.cat=mol['cat'].to(torch.long).view(-1, 1) 
                # data.time=mol['time']

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)
                
            pbar.close()
            torch.save(collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

        print(0)
