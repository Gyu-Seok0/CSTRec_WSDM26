from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import set_random_seed

class CustomDataset(Dataset):
    def __init__(self, data_instances):
        self.data_instances = data_instances
    
    def __len__(self):
        return len(self.data_instances)
    
    def __getitem__(self, idx):
        user_id, train_seq, pos_item_id, valid_item_id, test_item_id = self.data_instances[idx]
        return torch.tensor(user_id), torch.tensor(train_seq), torch.tensor(pos_item_id), torch.tensor(valid_item_id), torch.tensor(test_item_id)


class SeqRecDataset(Dataset):
    def __init__(self, block, window_size, target_size, num_train_neg, num_eval_neg, mode = "Train", eval_mode = "Full", data = "ml-1m", random_seed = 0):
        super(SeqRecDataset, self).__init__()
        
        set_random_seed(random_seed)
        
        self.User = defaultdict(list) # {user_id : [item_id, ...]}
        self.User_subseqs = defaultdict(list) # {user_id : [(train_seq, pos_item_id, valid_item_id, test_item_id), (train_seq, pos_item_id, valid_item_id, test_item_id), ...]}
        self.data_instances = [] # [(user_id, train_seq, pos_item_id, valid_item_id, test_item_id), ... ]
        self.train_seq_items = set() # all item ids in train seqs
        self.neg_train_seqs = [] # |seq| x num_train_neg x window_size
        
        self.window_size = window_size
        self.target_size = target_size
        self.max_length = window_size + target_size + 2 # 2 means valid and test items.
        self.num_train_neg = num_train_neg
        self.num_eval_neg = num_eval_neg
        self.mode = mode
        self.eval_mode = eval_mode
        self.data = data
        
        if block is not None:
            self._preprocess(block)
            
        # if data == "ml-1m":
        #     self.neg_sampling_train = self.neg_sampling_train_cpu
        # elif data == "gowalla":
        #     self.neg_sampling_train = self.neg_sampling_train_gpu
        self.neg_sampling_train = self.neg_sampling_train_gpu
        
    def _preprocess(self, block):
        
        for row in block.itertuples():
            self.User[row.user_id].append(row.item_id)
        
        '''Split sequences into train/valid/test'''
        for user_id in self.User:
            user_seq = self.User[user_id]
            user_seq_length = len(user_seq)
        
            if user_seq_length < 3:
                continue
            
            elif user_seq_length <= self.max_length:
                train_seq, pos_item_id, valid_item_id, test_item_id = self.get_train_test_valid_split(user_seq)
                if len(train_seq) == 0:
                    continue
                train_seq = self.make_zero_padding(train_seq)
                self.User_subseqs[user_id].append((train_seq, pos_item_id, valid_item_id, test_item_id))
                self.train_seq_items.update(train_seq)
            else:
                ''' Make subsequences '''
                for start in range(user_seq_length - self.max_length + 1):
                    end = start + self.max_length
                    train_seq, pos_item_id, valid_item_id, test_item_id = self.get_train_test_valid_split(user_seq[start : end])
                    if len(train_seq) == 0:
                        continue
                    train_seq = self.make_zero_padding(train_seq)
                    self.User_subseqs[user_id].append((train_seq, pos_item_id, valid_item_id, test_item_id))
                    self.train_seq_items.update(train_seq)
            
            #self.train_seq_items = self.train_seq_items.union(set(train_seq))
        
        ''' Valid/test items should be seen in the other users' train seq '''
        for user_id in self.User_subseqs.keys():
            for idx, user_data in enumerate(self.User_subseqs[user_id]):
                train_seq, pos_item_id, valid_item_id, test_item_id = user_data
                
                if valid_item_id not in self.train_seq_items or test_item_id not in self.train_seq_items:
                    train_seq = train_seq[2:] + [pos_item_id] + [valid_item_id]
                    pos_item_id = test_item_id
                    valid_item_id = test_item_id = 0
                    self.User_subseqs[user_id][idx] = (train_seq, pos_item_id, valid_item_id, test_item_id)
                    self.train_seq_items.update(train_seq)
                
                self.data_instances.append((user_id, train_seq, pos_item_id, valid_item_id, test_item_id))
                    
    def get_train_test_valid_split(self, user_seq):
        pos_item_id, valid_item_id, test_item_id = user_seq[-3], user_seq[-2], user_seq[-1]
        
        ''' Valid/test items should be unseen in the user's own train seq. '''
        train_seq = []
        for item_id in user_seq[:-3]:
            if item_id not in [valid_item_id, test_item_id]: 
                train_seq.append(item_id)
        return train_seq, pos_item_id, valid_item_id, test_item_id
    
    def make_zero_padding(self, seq):
        ''' Make zero padding to match the window size '''
        pad_seq = np.zeros([self.window_size], dtype=np.int32)
        pad_seq[-len(seq):] = seq
        return pad_seq.tolist()
    
    def neg_sampling_train_cpu(self):
        self.neg_train_seqs = []
        for data in tqdm(self.data_instances):
            user_id, train_seq, pos_item_id, valid_item_id, test_item_id = data
            seen_items = set(train_seq).union({pos_item_id, valid_item_id, test_item_id})
            neg_can = np.array(list(self.train_seq_items.difference(seen_items)))
            
            if len(neg_can) > 0:
                neg_seqs = np.random.choice(neg_can, (self.num_train_neg, len(train_seq))).tolist() # Create all negative sequences at once using broadcasting
            else:
                neg_seqs = [[] for _ in range(self.num_train_neg)] # Handle the case where there are no negative candidates

            self.neg_train_seqs.append(neg_seqs)
    
    def neg_sampling_train_gpu(self, batch_size=256, num_workers=1):
        self.neg_train_seqs = []

        # Create a dataset and dataloader for batching
        dataset = CustomDataset(self.data_instances)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        all_items = torch.tensor(list(self.train_seq_items), device='cuda' if torch.cuda.is_available() else 'cpu')

        for batch in dataloader:
            batch_user_ids, batch_train_seqs, batch_pos_items, batch_valid_items, batch_test_items = batch

            # Move tensors to the same device as all_items
            batch_train_seqs = batch_train_seqs.to(all_items.device)
            batch_pos_items = batch_pos_items.to(all_items.device)
            batch_valid_items = batch_valid_items.to(all_items.device)
            batch_test_items = batch_test_items.to(all_items.device)

            # Concatenate seen items for the whole batch
            seen_items = torch.cat([
                batch_train_seqs, 
                batch_pos_items.unsqueeze(1), 
                batch_valid_items.unsqueeze(1), 
                batch_test_items.unsqueeze(1)
            ], dim=1)
            
            # Generate negative samples for the whole batch
            mask = ~torch.isin(all_items, seen_items)  # Exclude seen items from all_items
            neg_can = all_items[mask]  # Reshape based on batch size
            
            # neg_indices = torch.randint(low=0, high=neg_can.size(0), size=(batch_size, self.num_train_neg, batch_train_seqs.size(1)), device=all_items.device)
            # neg_seqs = neg_can[neg_indices] # B x N x L
            
            current_batch_size = batch_train_seqs.size(0)
            if neg_can.size(0) == 0:
                # Handle case where no negative candidates are available
                neg_seqs = torch.zeros((current_batch_size, self.num_train_neg, batch_train_seqs.size(1)), dtype=torch.long, device=all_items.device)
            else:
                neg_indices = torch.randint(low=0, high=neg_can.size(0), size=(current_batch_size, self.num_train_neg, batch_train_seqs.size(1)), device=all_items.device)
                neg_seqs = neg_can[neg_indices] # B x N x L
            
            self.neg_train_seqs.extend(neg_seqs.cpu().tolist())  # Append results
        assert len(self.neg_train_seqs) == len(self.data_instances), "Neg_train_seqs length does not match data_instances length."

                
    def neg_sampling_eval(self, seen_items):
        seen_items = set(seen_items)
        neg_can = np.array(list(self.train_seq_items.difference(seen_items)))
        neg_items = np.random.choice(neg_can, self.num_eval_neg).tolist()
        return neg_items
 
    def __len__(self):
        return len(self.data_instances)
    
    def __getitem__(self, idx):
        
        user_id, train_seq, pos_item_id, valid_item_id, test_item_id = self.data_instances[idx]
        
        if self.mode == "Train":
            assert self.neg_train_seqs != [], "Negative sampling is not called"
            pos_seq = torch.tensor(train_seq[1:] + [pos_item_id]) # seq should be assgined as torch.tensor()
            neg_seq = torch.tensor(self.neg_train_seqs[idx])
            train_seq = torch.tensor(train_seq)
            
            return {'user_id' : user_id, 'train_seq' : train_seq, 'pos_seq' : pos_seq, 'neg_seq' : neg_seq}
        else:
            if self.mode == "Valid":
                eval_seq = train_seq[1:] + [pos_item_id]
                target_item_id = valid_item_id
            
            elif self.mode == "Test":
                eval_seq = train_seq[2:] + [pos_item_id, valid_item_id]
                target_item_id = test_item_id
            
            if self.eval_mode == "Full":
                return {'user_id' : user_id, 'seq' : torch.tensor(eval_seq), 'target_item' : target_item_id}
            
            elif self.eval_mode == "LOO":
                seen_items = eval_seq + [target_item_id]
                neg_items = self.neg_sampling_eval(seen_items) # list()
                target_items = torch.tensor([target_item_id] + neg_items) # Note that target_item is located in 0-index.
                return {'user_id' : user_id, 'seq' : torch.tensor(eval_seq), 'target_item' : target_items}

class FullBatchDataset(SeqRecDataset):
    def __init__(self, prev_dataset, cur_dataset):
        
        self.block = None
        self.window_size = cur_dataset.window_size
        self.target_size = cur_dataset.target_size
        self.num_train_neg = cur_dataset.num_train_neg
        self.num_eval_neg = cur_dataset.num_eval_neg
        self.mode = cur_dataset.mode
        self.eval_mode = cur_dataset.eval_mode

        super().__init__(self.block, self.window_size, self.target_size, self.num_train_neg, self.num_eval_neg, self.mode, self.eval_mode)
        
        self.data_instances = prev_dataset.data_instances + cur_dataset.data_instances
        self.train_seq_items = prev_dataset.train_seq_items.union(cur_dataset.train_seq_items)
        self.User = defaultdict(list) # {user_id : [item_id, ...]}
        self.User_subseqs = defaultdict(list) # {user_id : [(train_seq, pos_item_id, valid_item_id, test_item_id), (train_seq, pos_item_id, valid_item_id, test_item_id), ...]}
        self.total_user_ids = set(list(prev_dataset.User.keys()) + list(cur_dataset.User.keys()))
        
        for user_id in self.total_user_ids:
            self.User[user_id] = prev_dataset.User[user_id] + cur_dataset.User[user_id]
            self.User_subseqs[user_id] = prev_dataset.User_subseqs[user_id] + cur_dataset.User_subseqs[user_id]

class NewUserDataset(SeqRecDataset):
    def __init__(self, cur_dataset, new_user_ids):
        self.block = None
        self.window_size = cur_dataset.window_size
        self.target_size = cur_dataset.target_size
        self.num_train_neg = cur_dataset.num_train_neg
        self.num_eval_neg = cur_dataset.num_eval_neg
        self.mode = cur_dataset.mode
        self.eval_mode = cur_dataset.eval_mode
        super().__init__(self.block, self.window_size, self.target_size, self.num_train_neg, self.num_eval_neg, self.mode, self.eval_mode)
        
        self.train_seq_items = cur_dataset.train_seq_items
        self.data_instances = []
        self.User = defaultdict(list)
        self.User_subseqs = defaultdict(list)
        
        for new_user_id in new_user_ids:
            self.User[new_user_id] = cur_dataset.User[new_user_id]
            self.User_subseqs[new_user_id] = cur_dataset.User_subseqs[new_user_id]
        
        for data in cur_dataset.data_instances:
            user_id = data[0]
            if user_id in new_user_ids:
                self.data_instances.append(data)