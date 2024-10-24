import time
from collections import defaultdict
import torch.optim as optim
import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN
import math 
import numpy as np
import wandb
from torch.autograd import Variable
from torchmetrics.classification import Accuracy
class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        C.is_dataparallel = 1
        C.start_iter_num = 0
        C.vocab_size = 100
        C.train_type='adv'
        C.prob_safe = 1
        C.prob_unsafe = 0
        C.dpo_model = None
        C.perform_co=False
        C.univ_adv=False
        
        # dataloder parameters
        
        C.save_iter = 1000
        C.train_evaluate_iter = 500
        C.val_evaluate_iter = 50
        C.val_iter=100
        C.test_evaluate_iter = 500
        C.save_path = './'
        C.optimizer_load_path = './'
        C.model_load_path = './'
        C.num_workers = 4
        C.tokenizer = {}
        C.pad_token='#'
        C.num_cap = 4
        C.task_tokens = []


        # optimizer parameters
        C.max_iters = None
        C.max_window_possible = 75
        C.decay_lr = 1
        C.batch_size = 96
        C.test_batch_size = 1000
        C.learning_rate = 5e-4
        C.warmup_iters = 2000
        C.lr_decay_iters = 2000
        C.min_lr = 1e-6
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 
        C.grad_norm_clip = 1.0
        C.max_train_iters = 100
        C.max_val_iters = 10
        C.max_test_iters = 1
        C.is_dpo = 0
        C.dpo_weight_safe = 0
        C.dpo_weight_unsafe = 0

        # Attack params
        C.eps_soft_prompt = 0.5
        C.eps_soft_paraphrase = 1.0
        C.attack_norm = 'fro'

        C.attack_adv = 1
        C.attack_adv_targeted = 1
        C.attack_jailbreak_mg_text = 1
        C.attack_jailbreak_mg_tokens = 0
        C.attack_jailbreak_co = 0
        C.threat_count_adv=1
        C.threat_pos_adv = -1
        C.adv_attack_norm = 'fro'
        C.adv_attack_iters = 10
        C.jail_mg_text_attack_iters= 10
        C.jail_mg_text_attack_norm = 'fro'
        C.jail_mg_text_frac = 0.1
        C.jail_mg_text_attack_type = None
        C.n_embd = 192
        return C

    def __init__(self, config, model, train_dataset, val_dataset,test_dataset_unsafe, test_dataset_safe, test_dataset_intermediate, test_dataset_all, test_dataset_duplicate, test_dataset_id_mg, test_dataset_ood_mg, test_dataset_id_jailbreak_direct, test_dataset_safe_jailbreak_direct, test_dataset_id_jailbreak_comp, test_dataset_safe_jailbreak_comp, test_dataset_duplicate2, test_dataset_safe_co=[], test_dataset_id_co=[]):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.test_dataset_unsafe = test_dataset_unsafe
        self.test_dataset_safe = test_dataset_safe
        self.test_dataset_intermediate = test_dataset_intermediate
        self.test_dataset_all = test_dataset_all
        self.test_dataset_duplicate = test_dataset_duplicate
        self.test_dataset_id_mg = test_dataset_id_mg
        self.test_dataset_ood_mg = test_dataset_ood_mg
        self.test_dataset_id_jailbreak_direct = test_dataset_id_jailbreak_direct
        self.test_dataset_safe_jailbreak_direct = test_dataset_safe_jailbreak_direct
        self.test_dataset_id_jailbreak_comp = test_dataset_id_jailbreak_comp
        self.test_dataset_safe_jailbreak_comp = test_dataset_safe_jailbreak_comp
        self.test_dataset_safe_co = test_dataset_safe_co
        self.test_dataset_id_co = test_dataset_id_co
        self.test_dataset_duplicate2=test_dataset_duplicate2

        self.callbacks = defaultdict(list)
        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        #print("running on device", self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.is_dataparallel = config.is_dataparallel

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def get_lr(self, it, config):
        # 1) linear warmup for warmup_iters steps
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config.lr_decay_iters:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    def run_pretrain(self):
        model, config = self.model, self.config

        # setup the optimizer
        if config.is_dataparallel==1:
            self.optimizer = model.module.configure_optimizers(config)
        else:
            self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=torch.utils.data.RandomSampler(self.val_dataset, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )

        test_loader_unsafe = DataLoader(
            self.test_dataset_unsafe,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_unsafe, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )



        test_loader_safe = DataLoader(
            self.test_dataset_safe,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_safe, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )



        test_loader_intermediate = DataLoader(
            self.test_dataset_intermediate,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_intermediate,  replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )


        test_loader_all = DataLoader(
            self.test_dataset_all,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_all, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )



        test_loader_duplicate = DataLoader(
            self.test_dataset_duplicate,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_duplicate, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )


        test_loader_id_mg = DataLoader(
            self.test_dataset_id_mg,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_id_mg, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )


        test_loader_duplicate2 = DataLoader(
            self.test_dataset_duplicate2,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_duplicate2, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )


        test_loader_ood_mg = DataLoader(
            self.test_dataset_ood_mg,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_ood_mg, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )

        test_loader_id_jailbreak_direct = DataLoader(
            self.test_dataset_id_jailbreak_direct,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_id_jailbreak_direct, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )

        test_loader_id_jailbreak_comp = DataLoader(
            self.test_dataset_id_jailbreak_comp,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_id_jailbreak_comp, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )

        test_loader_safe_jailbreak_direct = DataLoader(
            self.test_dataset_safe_jailbreak_direct,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_safe_jailbreak_direct, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )

        test_loader_safe_jailbreak_comp = DataLoader(
            self.test_dataset_safe_jailbreak_comp,
            sampler=torch.utils.data.RandomSampler(self.test_dataset_safe_jailbreak_comp, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
        )

        if config.perform_co==True:
            test_loader_safe_co = DataLoader(
                self.test_dataset_safe_co,
                sampler=torch.utils.data.RandomSampler(self.test_dataset_safe_co, replacement=False),
                shuffle=False,
                pin_memory=True,
                batch_size=config.test_batch_size,
                num_workers=config.num_workers,
            )

            test_loader_id_co = DataLoader(
                self.test_dataset_id_co,
                sampler=torch.utils.data.RandomSampler(self.test_dataset_id_co, replacement=False),
                shuffle=False,
                pin_memory=True,
                batch_size=config.test_batch_size,
                num_workers=config.num_workers,
            )


        if config.model_load_path !='':
            model.load_state_dict(torch.load(config.model_load_path))

        if config.optimizer_load_path !='':
            optimizer.load_state_dict(torch.load(config.optimizer_load_path))


        model.train()
        self.iter_num = config.start_iter_num
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            lr = self.get_lr(self.iter_num, config) if config.decay_lr else config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, x_safe_cap_padded, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, y_safe_cap_paded, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch

            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()
            x_safe_cap_target = x_safe_cap_target.cuda()

            
            mask = mask.cuda()
            
            # forward the model
            if self.config.train_type=='adv':
                random_pick = np.random.choice([0,1],p=(config.prob_unsafe, config.prob_safe))
            elif self.config.train_config == 'filter':
                random_pick = 1
            
            if random_pick==0:
                x = x_unsafe_cap
                y = y_unsafe_cap
                if self.config.is_dpo==1:
                    x_ex = x_unsafe_cap_target
                    y_ex = y_unsafe_cap_target
                    dummy_mask_safe = 1 - (y_safe_cap_target == self.config.tokenizer[self.config.pad_token]).int()
                    dummy_mask_unsafe = 1 - (y_unsafe_cap_target == self.config.tokenizer[self.config.pad_token]).int()

                    conj_dummy_mask_safe = (y_safe_cap_target == self.config.tokenizer[self.config.pad_token]).int()
                    conj_dummy_mask_unsafe = (y_unsafe_cap_target == self.config.tokenizer[self.config.pad_token]).int()

            elif random_pick==1:
                x = x_safe_cap
                y = y_safe_cap
                if self.config.is_dpo==1:
                    x_ex = x_safe_cap_target
                    y_ex = y_safe_cap_target
                    dummy_mask_safe = 1 - (y_safe_cap_target == self.config.tokenizer[self.config.pad_token]).int()
                    dummy_mask_unsafe = 1 - (y_unsafe_cap_target == self.config.tokenizer[self.config.pad_token]).int()

                    conj_dummy_mask_safe = (y_safe_cap_target == self.config.tokenizer[self.config.pad_token]).int()
                    conj_dummy_mask_unsafe = (y_unsafe_cap_target == self.config.tokenizer[self.config.pad_token]).int()

            elif random_pick==2:
                x = x_unsafe_cap_target
                y = y_unsafe_cap_target

            
            if self.config.is_dpo==1:
                logits, self.loss_safe = model(x,y,mask,is_dpo=True)
                self.loss_safe = torch.mul(torch.reshape(self.loss_safe, [y.shape[0], y.shape[1]]), dummy_mask_safe)

                shape_value = self.loss_safe.shape[-1]
                self.loss_safe = torch.sum(self.loss_safe, dim=-1)/shape_value


                logits_unsafe, self.loss_unsafe = model(x_ex,y_ex,mask,is_dpo=True)
                self.loss_unsafe = torch.mul(torch.reshape(self.loss_unsafe, [y.shape[0], y.shape[1]]), dummy_mask_unsafe)
                
                shape_value = self.loss_unsafe.shape[-1]
                self.loss_unsafe = torch.sum(self.loss_unsafe, dim=-1)/shape_value


                #logits_dpo, self.loss_safe_dpo = config.dpo_model(x,y,mask,is_dpo=True)
                #self.loss_safe_dpo = torch.mul(torch.reshape(self.loss_safe_dpo, [y.shape[0], y.shape[1]]), dummy_mask_safe)
                #shape_value = self.loss_safe_dpo.shape[-1]
                #self.loss_safe_dpo = torch.sum(self.loss_safe_dpo, dim=-1)/shape_value

                #logits_unsafe_dpo, self.loss_unsafe_dpo = config.dpo_model(x_ex,y_ex,mask,is_dpo=True)
                #self.loss_unsafe_dpo = torch.mul(torch.reshape(self.loss_unsafe_dpo, [y.shape[0], y.shape[1]]), dummy_mask_unsafe)

                #shape_value = self.loss_unsafe_dpo.shape[-1]
                #self.loss_unsafe_dpo = torch.sum(self.loss_unsafe_dpo, dim=-1)/shape_value

                #print("loss_safe_dpo", self.loss_safe_dpo)
                #print("loss_unsafe_dpo", self.loss_unsafe_dpo)
                #print("loss_safe", self.loss_safe)
                #print("loss_unsafe", self.loss_unsafe)
                #a+=1




                # logits_dpo, self.loss_safe_dpo = config.dpo_model(x,y,mask,is_dpo=True)
                # logits_unsafe_dpo, self.loss_unsafe_dpo = config.dpo_model(x_ex,y_ex,mask,is_dpo=True)
                # self.loss_safe_dpo = torch.mul(torch.reshape(self.loss_safe_dpo, [y.shape[0], y.shape[1]]),dummy_mask_safe)
                # self.loss_unsafe_dpo = torch.mul(torch.reshape(self.loss_unsafe_dpo, [y.shape[0], y.shape[1]]),dummy_mask_unsafe)
                # self.loss_safe_dpo = torch.mean(self.loss_safe_dpo, dim=-1)
                # self.loss_unsafe_dpo = torch.mean(self.loss_unsafe_dpo, dim=-1)

                #self.loss = -torch.mean(((self.loss_safe_dpo.detach()-self.loss_safe)*self.config.dpo_weight_safe - (self.loss_unsafe_dpo.detach()-self.loss_unsafe)*self.config.dpo_weight_unsafe))
                self.loss = torch.mean(self.loss_safe*self.config.dpo_weight_safe - self.loss_unsafe*self.config.dpo_weight_unsafe)

            
            else:
                logits, self.loss = model(x,y,mask)

            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            logits_mod = []
            target_label = []
            acc_temp = 0
            acc_patch = 0
            counter_iter = 0

            logits = logits.cuda()
            #torch.set_printoptions(threshold=10_000)
            y = y.cuda()
            #print("Reshaped",torch.reshape(logits,[-1,62]).permute(1,0).shape)
            #print("Y shape", y.view(-1).shape)
            #print("Pred",logits.argmax(dim=-1).detach().cpu())
            #print("y label", y.detach().cpu())

            acc_comb1_safebranch_safecap=0
            acc_comb2_safebranch_safecap=0
            acc_comb_safebranch_safecap=0
            counter_safebranch_safecap1=0
            counter_safebranch_safecap2=0
            counter_safebranch_safecap12=0
            acc_safebranch_safecap=0
            #acc_pcfg_safebranch_safecap=0
            acc_idx1_safebranch_safecap=0
            acc_idx2_safebranch_safecap=0

            acc_comb1_safebranch_unsafecap=0
            acc_comb2_safebranch_unsafecap=0
            acc_comb_safebranch_unsafecap=0
            counter_safebranch_unsafecap12=0
            counter_safebranch_unsafecap2=0
            counter_safebranch_unsafecap1=0
            counter_safebranch_unsafecap12_tar=0
            counter_safebranch_unsafecap2_tar=0
            counter_safebranch_unsafecap1_tar=0
            acc_safebranch_unsafecap=0
            #acc_pcfg_safebranch_unsafecap=0
            acc_idx1_safebranch_unsafecap=0
            acc_idx2_safebranch_unsafecap=0


            acc_comb1_safebranch_unsafecap_tar=0
            acc_comb2_safebranch_unsafecap_tar=0
            acc_comb_safebranch_unsafecap_tar=0
            acc_safebranch_unsafecap_tar=0
            #acc_pcfg_safebranch_unsafecap=0
            acc_idx1_safebranch_unsafecap_tar=0
            acc_idx2_safebranch_unsafecap_tar=0

            acc_comb1_unsafebranch_safecap=0
            acc_comb2_unsafebranch_safecap=0
            acc_comb_unsafebranch_safecap=0
            counter_unsafebranch_safecap12=0
            counter_unsafebranch_safecap1=0
            counter_unsafebranch_safecap2=0

            acc_unsafebranch_safecap=0
            #acc_pcfg_unsafebranch_safecap=0
            acc_idx1_unsafebranch_safecap=0
            acc_idx2_unsafebranch_safecap=0

            acc_comb1_unsafebranch_unsafecap=0
            acc_comb2_unsafebranch_unsafecap=0
            acc_comb_unsafebranch_unsafecap=0
            counter_unsafebranch_unsafecap2=0
            counter_unsafebranch_unsafecap1=0
            counter_unsafebranch_unsafecap12=0
            counter_unsafebranch_unsafecap2_tar=0
            counter_unsafebranch_unsafecap1_tar=0
            counter_unsafebranch_unsafecap12_tar=0
            acc_unsafebranch_unsafecap=0
            #acc_pcfg_unsafebranch_unsafecap=0
            acc_idx1_unsafebranch_unsafecap=0
            acc_idx2_unsafebranch_unsafecap=0

            acc_comb1_unsafebranch_unsafecap_tar=0
            acc_comb2_unsafebranch_unsafecap_tar=0
            acc_comb_unsafebranch_unsafecap_tar=0
            acc_unsafebranch_unsafecap_tar=0
            #acc_pcfg_unsafebranch_unsafecap=0
            acc_idx1_unsafebranch_unsafecap_tar=0
            acc_idx2_unsafebranch_unsafecap_tar=0


            for i in range(y.shape[0]):

                if safety_label[i]==1 and random_pick==1:
                    if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token] and y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_temp = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][-1]])
                        acc_safebranch_safecap+=torch.sum(acc_temp/acc_temp.shape[0]).item()
                        counter_safebranch_safecap12+=1
                        if torch.sum(acc_temp) == (end_idx[i][-1]-start_idx[i]):
                            acc_comb_safebranch_safecap+=1



                    if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token]:
                        counter_safebranch_safecap1+=1
                        acc_idx1_temp = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][0]])
                        acc_idx1_safebranch_safecap+=torch.sum(acc_idx1_temp/acc_idx1_temp.shape[0]).item()
                        if torch.sum(acc_idx1_temp)==(end_idx[i][0]-start_idx[i]):
                            acc_comb1_safebranch_safecap+=1

                    if y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        counter_safebranch_safecap2+=1
                        acc_idx2_temp = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y[i][end_idx[i][0]:end_idx[i][1]])                    
                        acc_idx2_safebranch_safecap+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()
                        if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                            acc_comb2_safebranch_safecap+=1





                if safety_label[i]==1 and random_pick==0:
                    if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token] and y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_temp = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][-1]])
                        acc_safebranch_unsafecap+=torch.sum(acc_temp/acc_temp.shape[0]).item()
                        if torch.sum(acc_temp) == (end_idx[i][-1]-start_idx[i]):
                            acc_comb_safebranch_unsafecap+=1
                        counter_safebranch_unsafecap12+=1

                    if y_unsafe_cap_target[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token] and y_unsafe_cap_target[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_temp2 = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y_unsafe_cap_target[i][start_idx[i]:end_idx[i][-1]])
                        acc_safebranch_unsafecap_tar+=torch.sum(acc_temp2/acc_temp2.shape[0]).item()
                        if torch.sum(acc_temp2) == (end_idx[i][-1]-start_idx[i]):
                            acc_comb_safebranch_unsafecap_tar+=1
                        counter_safebranch_unsafecap12_tar+=1
                    

                    


                    #acc_temp_pcfg = (logits[i][1+config.num_cap:start_idx[i]].argmax(dim=-1)==y[i][1+config.num_cap:start_idx[i]])
                    #acc_pcfg_safebranch_unsafecap+=torch.sum(acc_temp_pcfg/acc_temp_pcfg.shape[0]).item()

                    if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_idx1_temp = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][0]])
                        acc_idx1_safebranch_unsafecap+=torch.sum(acc_idx1_temp/acc_idx1_temp.shape[0]).item()

                        if torch.sum(acc_idx1_temp)==(end_idx[i][0]-start_idx[i]):
                            acc_comb1_safebranch_unsafecap+=1
                        counter_safebranch_unsafecap1+=1

                    if y_unsafe_cap_target[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_idx1_temp2 = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y_unsafe_cap_target[i][start_idx[i]:end_idx[i][0]])
                        acc_idx1_safebranch_unsafecap_tar+=torch.sum(acc_idx1_temp2/acc_idx1_temp2.shape[0]).item()


                        if torch.sum(acc_idx1_temp2)==(end_idx[i][0]-start_idx[i]):
                            acc_comb1_safebranch_unsafecap_tar+=1
                        
                        counter_safebranch_unsafecap1_tar+=1


                    if y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_idx2_temp = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y[i][end_idx[i][0]:end_idx[i][1]])                    
                        acc_idx2_safebranch_unsafecap+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()

                        if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                            acc_comb2_safebranch_unsafecap+=1
                        counter_safebranch_unsafecap2+=1

                    if y_unsafe_cap_target[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_idx2_temp2 = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y_unsafe_cap_target[i][end_idx[i][0]:end_idx[i][1]])                    
                        acc_idx2_safebranch_unsafecap_tar+=torch.sum(acc_idx2_temp2/acc_idx2_temp2.shape[0]).item()

                
                        if torch.sum(acc_idx2_temp2)==(end_idx[i][1]-end_idx[i][0]):
                            acc_comb2_safebranch_unsafecap_tar+=1
                        counter_safebranch_unsafecap2_tar+=1



                if (safety_label[i]==0 or safety_label[i]==5)  and random_pick==1:
                    if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token] and y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_temp = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][-1]])
                        acc_unsafebranch_safecap+=torch.sum(acc_temp/acc_temp.shape[0]).item()
                        counter_unsafebranch_safecap12+=1
                        if torch.sum(acc_temp) == (end_idx[i][-1]-start_idx[i]):
                            acc_comb_unsafebranch_safecap+=1


                    if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token]:
                        counter_unsafebranch_safecap1+=1
                        acc_idx1_temp = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][0]])
                        acc_idx1_unsafebranch_safecap+=torch.sum(acc_idx1_temp/acc_idx1_temp.shape[0]).item()
                        if torch.sum(acc_idx1_temp)==(end_idx[i][0]-start_idx[i]):
                            acc_comb1_unsafebranch_safecap+=1


                    if y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        counter_unsafebranch_safecap2+=1
                        acc_idx2_temp = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y[i][end_idx[i][0]:end_idx[i][1]])                    
                        acc_idx2_unsafebranch_safecap+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()
                        if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                            acc_comb2_unsafebranch_safecap+=1
                



                if (safety_label[i]==0 or safety_label[i]==5) and random_pick==0:
                    if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token] and y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        acc_temp = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][-1]])
                        counter_unsafebranch_unsafecap12+=1
                        if torch.sum(acc_temp) == (end_idx[i][-1]-start_idx[i]):
                            acc_comb_unsafebranch_unsafecap+=1
                        acc_unsafebranch_unsafecap+=torch.sum(acc_temp/acc_temp.shape[0]).item()

                    if y_unsafe_cap_target[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token] and y_unsafe_cap_target[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        counter_unsafebranch_unsafecap12_tar+=1

                        acc_temp2 = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y_unsafe_cap_target[i][start_idx[i]:end_idx[i][-1]])

                        
                        acc_unsafebranch_unsafecap_tar+=torch.sum(acc_temp2/acc_temp2.shape[0]).item()


                    
                        if torch.sum(acc_temp2) == (end_idx[i][-1]-start_idx[i]):
                            acc_comb_unsafebranch_unsafecap_tar+=1

                    #acc_temp_pcfg = (logits[i][1+config.num_cap:start_idx[i]].argmax(dim=-1)==y[i][1+config.num_cap:start_idx[i]])
                    #acc_pcfg_unsafebranch_unsafecap+=torch.sum(acc_temp_pcfg/acc_temp_pcfg.shape[0]).item()

                    if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token]:
                        counter_unsafebranch_unsafecap1+=1
                        acc_idx1_temp = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][0]])
                        acc_idx1_unsafebranch_unsafecap+=torch.sum(acc_idx1_temp/acc_idx1_temp.shape[0]).item()
                        if torch.sum(acc_idx1_temp)==(end_idx[i][0]-start_idx[i]):
                            acc_comb1_unsafebranch_unsafecap+=1
                        
                    if y_unsafe_cap_target[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token]: 
                        counter_unsafebranch_unsafecap1_tar+=1
                        acc_idx1_temp2 = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y_unsafe_cap_target[i][start_idx[i]:end_idx[i][0]])
                        acc_idx1_unsafebranch_unsafecap_tar+=torch.sum(acc_idx1_temp2/acc_idx1_temp2.shape[0]).item()

                        if torch.sum(acc_idx1_temp2)==(end_idx[i][0]-start_idx[i]):
                            acc_comb1_unsafebranch_unsafecap_tar+=1



                    if y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        counter_unsafebranch_unsafecap2+=1
                        acc_idx2_temp = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y[i][end_idx[i][0]:end_idx[i][1]])                    
                        acc_idx2_unsafebranch_unsafecap+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()

                        if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                            acc_comb2_unsafebranch_unsafecap+=1
                    
                    if y_unsafe_cap_target[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                        counter_unsafebranch_unsafecap2_tar+=1
                        acc_idx2_temp2 = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y_unsafe_cap_target[i][end_idx[i][0]:end_idx[i][1]])                    
                        acc_idx2_unsafebranch_unsafecap_tar+=torch.sum(acc_idx2_temp2/acc_idx2_temp2.shape[0]).item()

                        if torch.sum(acc_idx2_temp2)==(end_idx[i][1]-end_idx[i][0]):
                            acc_comb2_unsafebranch_unsafecap_tar+=1


            try:
                size_lst = self.loss.shape[0]
                self.loss = torch.sum(self.loss)/self.loss.shape[0]
            except:
                self.loss = self.loss


            if counter_safebranch_safecap12==0:
                counter_safebranch_safecap12=1
            if counter_safebranch_unsafecap12==0:
                counter_safebranch_unsafecap12=1
            if counter_unsafebranch_unsafecap12==0:
                counter_unsafebranch_unsafecap12=1
            if counter_unsafebranch_safecap12==0:
                counter_unsafebranch_safecap12=1

            if counter_safebranch_safecap1==0:
                counter_safebranch_safecap1=1
            if counter_safebranch_unsafecap1==0:
                counter_safebranch_unsafecap1=1
            if counter_unsafebranch_unsafecap1==0:
                counter_unsafebranch_unsafecap1=1
            if counter_unsafebranch_safecap1==0:
                counter_unsafebranch_safecap1=1

            if counter_safebranch_safecap2==0:
                counter_safebranch_safecap2=1
            if counter_safebranch_unsafecap2==0:
                counter_safebranch_unsafecap2=1
            
            if counter_safebranch_unsafecap2_tar==0:
                counter_safebranch_unsafecap2_tar=1
            if counter_unsafebranch_unsafecap2_tar==0:
                counter_unsafebranch_unsafecap2_tar=1
                
            if counter_unsafebranch_unsafecap2==0:
                counter_unsafebranch_unsafecap2=1
            if counter_unsafebranch_safecap2==0:
                counter_unsafebranch_safecap2=1

            if counter_safebranch_unsafecap1_tar==0:
                counter_safebranch_unsafecap1_tar=1
            if counter_unsafebranch_unsafecap1_tar==0:
                counter_unsafebranch_unsafecap1_tar=1

            if counter_safebranch_unsafecap12_tar==0:
                counter_safebranch_unsafecap12_tar=1
            if counter_unsafebranch_unsafecap12_tar==0:
                counter_unsafebranch_unsafecap12_tar=1


            acc_iter_safebranch_safecap = acc_safebranch_safecap/counter_safebranch_safecap12
            acc_idx1_iter_safebranch_safecap = acc_idx1_safebranch_safecap/counter_safebranch_safecap1
            acc_idx2_iter_safebranch_safecap = acc_idx2_safebranch_safecap/counter_safebranch_safecap2
            acc_comb1_iter_safebranch_safecap = acc_comb1_safebranch_safecap/counter_safebranch_safecap1
            acc_comb2_iter_safebranch_safecap = acc_comb2_safebranch_safecap/counter_safebranch_safecap2
            acc_comb_iter_safebranch_safecap = acc_comb_safebranch_safecap/counter_safebranch_safecap12



            acc_iter_safebranch_unsafecap = acc_safebranch_unsafecap/counter_safebranch_unsafecap12
            acc_idx1_iter_safebranch_unsafecap = acc_idx1_safebranch_unsafecap/counter_safebranch_unsafecap1
            acc_idx2_iter_safebranch_unsafecap = acc_idx2_safebranch_unsafecap/counter_safebranch_unsafecap2
            acc_comb1_iter_safebranch_unsafecap = acc_comb1_safebranch_unsafecap/counter_safebranch_unsafecap1
            acc_comb2_iter_safebranch_unsafecap = acc_comb2_safebranch_unsafecap/counter_safebranch_unsafecap2
            acc_comb_iter_safebranch_unsafecap = acc_comb_safebranch_unsafecap/counter_safebranch_unsafecap12


            acc_iter_unsafebranch_unsafecap = acc_unsafebranch_unsafecap/counter_unsafebranch_unsafecap12
            acc_idx1_iter_unsafebranch_unsafecap = acc_idx1_unsafebranch_unsafecap/counter_unsafebranch_unsafecap1
            acc_idx2_iter_unsafebranch_unsafecap = acc_idx2_unsafebranch_unsafecap/counter_unsafebranch_unsafecap2
            acc_comb1_iter_unsafebranch_unsafecap = acc_comb1_unsafebranch_unsafecap/counter_unsafebranch_unsafecap1
            acc_comb2_iter_unsafebranch_unsafecap = acc_comb2_unsafebranch_unsafecap/counter_unsafebranch_unsafecap2
            acc_comb_iter_unsafebranch_unsafecap = acc_comb_unsafebranch_unsafecap/counter_unsafebranch_unsafecap12


            acc_iter_safebranch_unsafecap_tar = acc_safebranch_unsafecap_tar/counter_safebranch_unsafecap12_tar
            acc_idx1_iter_safebranch_unsafecap_tar = acc_idx1_safebranch_unsafecap_tar/counter_safebranch_unsafecap1_tar
            acc_idx2_iter_safebranch_unsafecap_tar = acc_idx2_safebranch_unsafecap_tar/counter_safebranch_unsafecap2_tar
            acc_comb1_iter_safebranch_unsafecap_tar = acc_comb1_safebranch_unsafecap_tar/counter_safebranch_unsafecap1_tar
            acc_comb2_iter_safebranch_unsafecap_tar = acc_comb2_safebranch_unsafecap_tar/counter_safebranch_unsafecap2_tar
            acc_comb_iter_safebranch_unsafecap_tar = acc_comb_safebranch_unsafecap_tar/counter_safebranch_unsafecap12_tar


            acc_iter_unsafebranch_unsafecap_tar = acc_unsafebranch_unsafecap_tar/counter_unsafebranch_unsafecap12_tar
            acc_idx1_iter_unsafebranch_unsafecap_tar = acc_idx1_unsafebranch_unsafecap_tar/counter_unsafebranch_unsafecap1_tar
            acc_idx2_iter_unsafebranch_unsafecap_tar = acc_idx2_unsafebranch_unsafecap_tar/counter_unsafebranch_unsafecap2_tar
            acc_comb1_iter_unsafebranch_unsafecap_tar = acc_comb1_unsafebranch_unsafecap_tar/counter_unsafebranch_unsafecap1_tar
            acc_comb2_iter_unsafebranch_unsafecap_tar = acc_comb2_unsafebranch_unsafecap_tar/counter_unsafebranch_unsafecap2_tar
            acc_comb_iter_unsafebranch_unsafecap_tar = acc_comb_unsafebranch_unsafecap_tar/counter_unsafebranch_unsafecap12_tar

 
            acc_iter_unsafebranch_safecap = acc_unsafebranch_safecap/counter_unsafebranch_safecap12
            acc_idx1_iter_unsafebranch_safecap = acc_idx1_unsafebranch_safecap/counter_unsafebranch_safecap1
            acc_idx2_iter_unsafebranch_safecap = acc_idx2_unsafebranch_safecap/counter_unsafebranch_safecap2
            acc_comb1_iter_unsafebranch_safecap = acc_comb1_unsafebranch_safecap/counter_unsafebranch_safecap1
            acc_comb2_iter_unsafebranch_safecap = acc_comb2_unsafebranch_safecap/counter_unsafebranch_safecap2
            acc_comb_iter_unsafebranch_safecap = acc_comb_unsafebranch_safecap/counter_unsafebranch_safecap12


            if random_pick==1:
                wandb.log({'Loss (Train)': self.loss.item()},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (Sfe Cp) Cp1,2 (Train)': acc_iter_safebranch_safecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (Sfe Cp) Cp 1 (Train)': acc_idx1_iter_safebranch_safecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (Sfe Cp) Cp 2 (Train)': acc_idx2_iter_safebranch_safecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (Sfe Cp) Cp 1 And (Train)': acc_comb1_iter_safebranch_safecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (Sfe Cp) Cp 2 And (Train)': acc_comb2_iter_safebranch_safecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (Sfe Cp) Cp 1,2 And (Train)': acc_comb_iter_safebranch_safecap},step=self.iter_num)

                wandb.log({'Acc USfe Brch (Sfe Cp) Cp1,2 (Train)': acc_iter_unsafebranch_safecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (Sfe Cp) Cp 1 (Train)': acc_idx1_iter_unsafebranch_safecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (Sfe Cp) Cp 2 (Train)': acc_idx2_iter_unsafebranch_safecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (Sfe Cp) Cp 1 And (Train)': acc_comb1_iter_unsafebranch_safecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (Sfe Cp) Cp 2 And (Train)': acc_comb2_iter_unsafebranch_safecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (Sfe Cp) Cp 1,2 And (Train)': acc_comb_iter_unsafebranch_safecap},step=self.iter_num)



            elif random_pick==0:
                wandb.log({'Acc Sfe Brch (USfe Cp) Cp1,2 (Train)': acc_iter_safebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Cp 1 (Train)': acc_idx1_iter_safebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Cp 2 (Train)': acc_idx2_iter_safebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Cp 1 And (Train)': acc_comb1_iter_safebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Cp 2 And (Train)': acc_comb2_iter_safebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Cp 1,2 And (Train)': acc_comb_iter_safebranch_unsafecap},step=self.iter_num)

                wandb.log({'Acc USfe Brch (USfe Cp) Cp1,2 (Train)': acc_iter_unsafebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Cp 1 (Train)': acc_idx1_iter_unsafebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Cp 2 (Train)': acc_idx2_iter_unsafebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Cp 1 And (Train)': acc_comb1_iter_unsafebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Cp 2 And (Train)': acc_comb2_iter_unsafebranch_unsafecap},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Cp 1,2 And (Train)': acc_comb_iter_unsafebranch_unsafecap},step=self.iter_num)



                wandb.log({'Acc Sfe Brch (USfe Cp) Targeted Cp1,2 (Train)': acc_iter_safebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Targeted Cp 1 (Train)': acc_idx1_iter_safebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Targeted Cp 2 (Train)': acc_idx2_iter_safebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Targeted Cp 1 And (Train)': acc_comb1_iter_safebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Targeted Cp 2 And (Train)': acc_comb2_iter_safebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc Sfe Brch (USfe Cp) Targeted Cp 1,2 And (Train)': acc_comb_iter_safebranch_unsafecap_tar},step=self.iter_num)

                wandb.log({'Acc USfe Brch (USfe Cp) Targeted Cp1,2 (Train)': acc_iter_unsafebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Targeted Cp 1 (Train)': acc_idx1_iter_unsafebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Targeted Cp 2 (Train)': acc_idx2_iter_unsafebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Targeted Cp 1 And (Train)': acc_comb1_iter_unsafebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Targeted Cp 2 And (Train)': acc_comb2_iter_unsafebranch_unsafecap_tar},step=self.iter_num)
                wandb.log({'Acc USfe Brch (USfe Cp) Targeted Cp 1,2 And (Train)': acc_comb_iter_unsafebranch_unsafecap_tar},step=self.iter_num)




            if self.iter_num%config.test_evaluate_iter==0 or self.iter_num==1: 
                print("Train iter {} Loss {}".format(self.iter_num, self.loss.item()))
                #print("Train iter {} Accuracy Next Token PCFG {}".format(self.iter_num,acc_iter_pcfg))
                print("Train SS iter {} Accuracy Cp1,2 {}".format(self.iter_num,acc_iter_safebranch_safecap))
                print("Train SS iter {} Accuracy Cp 1 {}".format(self.iter_num, acc_idx1_iter_safebranch_safecap))
                print("Train SS iter {} Accuracy Cp 2 {}".format(self.iter_num, acc_idx2_iter_safebranch_safecap))
                print("Train SS iter {} Accuracy Cp 1 And {}".format(self.iter_num, acc_comb1_iter_safebranch_safecap))
                print("Train SS iter {} Accuracy Cp 2 And {}".format(self.iter_num, acc_comb2_iter_safebranch_safecap))
                print("Train SS iter {} Accuracy Cp 1,2 And {}".format(self.iter_num, acc_comb_iter_safebranch_safecap))

                #print("Train iter {} Accuracy Next Token PCFG {}".format(self.iter_num,acc_iter_pcfg))
                print("Train US iter {} Accuracy Cp1,2 {}".format(self.iter_num,acc_iter_unsafebranch_safecap))
                print("Train US iter {} Accuracy Cp 1 {}".format(self.iter_num, acc_idx1_iter_unsafebranch_safecap))
                print("Train US iter {} Accuracy Cp 2 {}".format(self.iter_num, acc_idx2_iter_unsafebranch_safecap))
                print("Train US iter {} Accuracy Cp 1 And {}".format(self.iter_num, acc_comb1_iter_unsafebranch_safecap))
                print("Train US iter {} Accuracy Cp 2 And {}".format(self.iter_num, acc_comb2_iter_unsafebranch_safecap))
                print("Train US iter {} Accuracy Cp 1,2 And {}".format(self.iter_num, acc_comb_iter_unsafebranch_safecap))


                #print("Train iter {} Accuracy Next Token PCFG {}".format(self.iter_num,acc_iter_pcfg))
                print("Train SU iter {} Accuracy Cp1,2 {}".format(self.iter_num,acc_iter_safebranch_unsafecap))
                print("Train SU iter {} Accuracy Cp 1 {}".format(self.iter_num, acc_idx1_iter_safebranch_unsafecap))
                print("Train SU iter {} Accuracy Cp 2 {}".format(self.iter_num, acc_idx2_iter_safebranch_unsafecap))
                print("Train SU iter {} Accuracy Cp 1 And {}".format(self.iter_num, acc_comb1_iter_safebranch_unsafecap))
                print("Train SU iter {} Accuracy Cp 2 And {}".format(self.iter_num, acc_comb2_iter_safebranch_unsafecap))
                print("Train SU iter {} Accuracy Cp 1,2 And {}".format(self.iter_num, acc_comb_iter_safebranch_unsafecap))

                #print("Train iter {} Accuracy Next Token PCFG {}".format(self.iter_num,acc_iter_pcfg))
                print("Train UU iter {} Accuracy Cp1,2 {}".format(self.iter_num,acc_iter_unsafebranch_unsafecap))
                print("Train UU iter {} Accuracy Cp 1 {}".format(self.iter_num, acc_idx1_iter_unsafebranch_unsafecap))
                print("Train UU iter {} Accuracy Cp 2 {}".format(self.iter_num, acc_idx2_iter_unsafebranch_unsafecap))
                print("Train UU iter {} Accuracy Cp 1 And {}".format(self.iter_num, acc_comb1_iter_unsafebranch_unsafecap))
                print("Train UU iter {} Accuracy Cp 2 And {}".format(self.iter_num, acc_comb2_iter_unsafebranch_unsafecap))
                print("Train UU iter {} Accuracy Cp 1,2 And {}".format(self.iter_num, acc_comb_iter_unsafebranch_unsafecap))

                #print("Train iter {} Accuracy Next Token PCFG {}".format(self.iter_num,acc_iter_pcfg))
                print("Train SU Tar iter {} Accuracy Cp1,2 {}".format(self.iter_num,acc_iter_safebranch_unsafecap_tar))
                print("Train SU Tar iter {} Accuracy Cp 1 {}".format(self.iter_num, acc_idx1_iter_safebranch_unsafecap_tar))
                print("Train SU Tar iter {} Accuracy Cp 2 {}".format(self.iter_num, acc_idx2_iter_safebranch_unsafecap_tar))
                print("Train SU Tar iter {} Accuracy Cp 1 And {}".format(self.iter_num, acc_comb1_iter_safebranch_unsafecap_tar))
                print("Train SU Tar iter {} Accuracy Cp 2 And {}".format(self.iter_num, acc_comb2_iter_safebranch_unsafecap_tar))
                print("Train SU Tar iter {} Accuracy Cp 1,2 And {}".format(self.iter_num, acc_comb_iter_safebranch_unsafecap_tar))

                #print("Train iter {} Accuracy Next Token PCFG {}".format(self.iter_num,acc_iter_pcfg))
                print("Train UU Tar iter {} Accuracy Cp1,2 {}".format(self.iter_num,acc_iter_unsafebranch_unsafecap_tar))
                print("Train UU Tar iter {} Accuracy Cp 1 {}".format(self.iter_num, acc_idx1_iter_unsafebranch_unsafecap_tar))
                print("Train UU Tar iter {} Accuracy Cp 2 {}".format(self.iter_num, acc_idx2_iter_unsafebranch_unsafecap_tar))
                print("Train UU Tar iter {} Accuracy Cp 1 And {}".format(self.iter_num, acc_comb1_iter_unsafebranch_unsafecap_tar))
                print("Train UU Tar iter {} Accuracy Cp 2 And {}".format(self.iter_num, acc_comb2_iter_unsafebranch_unsafecap_tar))
                print("Train UU Tar iter {} Accuracy Cp 1,2 And {}".format(self.iter_num, acc_comb_iter_unsafebranch_unsafecap_tar))



            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if self.iter_num%config.save_iter==0 and self.iter_num>0:
                torch.save(model.state_dict(),'safety_finetuned_models_new/' + config.save_path + '/model_{}.pkl'.format(self.iter_num))
            
            attack_lst = []
            if self.config.attack_adv==1:
                attack_lst.append('adv')
            if self.config.attack_adv_targeted==1:
                attack_lst.append('adv_target')
            if  self.config.attack_jailbreak_mg_text==1:
                attack_lst.append('jail_mg_text')
            if  self.config.attack_jailbreak_mg_tokens==1:
                attack_lst.append('jail_mg_tokens')
            if self.config.attack_jailbreak_co==1:
                attack_lst.append('jail_co')        



            if self.iter_num%config.test_evaluate_iter==0 or self.iter_num==1:   
                loss_iter_dict, acc_iter_dict, acc_pcfg_iter_dict, acc_idx1_iter_dict, acc_idx2_iter_dict, acc_comb1_iter_dict, acc_comb2_iter_dict, acc_comb_iter_dict = self.evaluate_pretrain(model, test_loader_safe, test_loader_unsafe, test_loader_intermediate, test_loader_all, test_loader_duplicate, test_loader_id_mg, test_loader_ood_mg, test_loader_id_jailbreak_comp, test_loader_safe_jailbreak_comp, test_loader_id_jailbreak_direct, test_loader_safe_jailbreak_direct,test_loader_duplicate2,  max_iterations=config.max_test_iters, attack_lst=attack_lst)

                wandb.log({'Loss Std Sfe Brch (Sfe Cp)(Tst)': loss_iter_dict['loss_safe_std'][0]},step=self.iter_num)
                wandb.log({'Loss Std Sfe Brch (USfe Cp)(Tst)': loss_iter_dict['loss_safe_std'][1]},step=self.iter_num)
                wandb.log({'Loss Std Sfe Brch (USfe Trg Cp)(Tst)': loss_iter_dict['loss_safe_std'][2]},step=self.iter_num)
                wandb.log({'Acc Next Token PCFG(Tst)': acc_pcfg_iter_dict['acc_safe_pcfg_std']},step=self.iter_num)

                wandb.log({'Acc Std Cp 1,2 Sfe Brch (Sfe Cp)(Tst)': acc_iter_dict['acc_safe_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1,2 Sfe Brch (USfe Cp)(Tst)': acc_iter_dict['acc_safe_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1,2 Sfe Brch (USfe Trg Cp)(Tst)': acc_iter_dict['acc_safe_std'][2]},step=self.iter_num)
            
                wandb.log({'Acc Std Cp 1 Sfe Brch (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_safe_idx1_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1 Sfe Brch (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_safe_idx1_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1 Sfe Brch (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_safe_idx1_std'][2]},step=self.iter_num)

                wandb.log({'Acc Std Cp 2 Sfe Brch (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_safe_idx2_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 2 Sfe Brch (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_safe_idx2_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 2 Sfe Brch (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_safe_idx2_std'][2]},step=self.iter_num)
    
                wandb.log({'Acc Std Cp 1 AND Sfe Brch (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_safe_comb1_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1 AND Sfe Brch (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_safe_comb1_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1 AND Sfe Brch (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_safe_comb1_std'][2]},step=self.iter_num)

                wandb.log({'Acc Std Cp 2 AND Sfe Brch (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_safe_comb2_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 2 AND Sfe Brch (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_safe_comb2_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 2 AND Sfe Brch (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_safe_comb2_std'][2]},step=self.iter_num)

                wandb.log({'Acc Std Cp 1,2 AND Sfe Brch (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_safe_comb_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1,2 AND Sfe Brch (USfe Cp)(Tst)': acc_comb_iter_dict['acc_safe_comb_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1,2 AND Sfe Brch (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_safe_comb_std'][2]},step=self.iter_num)
 



                wandb.log({'Loss Adv Sfe Brch (Sfe Cp)(Tst)': loss_iter_dict['loss_safe_adv'][0]},step=self.iter_num)
                wandb.log({'Loss Adv Sfe Brch (USfe Cp)(Tst)': loss_iter_dict['loss_safe_adv'][1]},step=self.iter_num)
                wandb.log({'Loss Adv Sfe Brch (USfe Trg Cp)(Tst)': loss_iter_dict['loss_safe_adv'][2]},step=self.iter_num)

                wandb.log({'Acc Adv Cp 1,2 Sfe Brch (Sfe Cp)(Tst)': acc_iter_dict['acc_safe_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1,2 Sfe Brch (USfe Cp)(Tst)': acc_iter_dict['acc_safe_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1,2 Sfe Brch (USfe Trg Cp)(Tst)': acc_iter_dict['acc_safe_adv'][2]},step=self.iter_num)
            
                wandb.log({'Acc Adv Cp 1 Sfe Brch (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_safe_idx1_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1 Sfe Brch (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_safe_idx1_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1 Sfe Brch (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_safe_idx1_adv'][2]},step=self.iter_num)

                wandb.log({'Acc Adv Cp 2 Sfe Brch (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_safe_idx2_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 2 Sfe Brch (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_safe_idx2_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 2 Sfe Brch (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_safe_idx2_adv'][2]},step=self.iter_num)
    
                wandb.log({'Acc Adv Cp 1 AND Sfe Brch (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_safe_comb1_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1 AND Sfe Brch (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_safe_comb1_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1 AND Sfe Brch (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_safe_comb1_adv'][2]},step=self.iter_num)

                wandb.log({'Acc Adv Cp 2 AND Sfe Brch (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_safe_comb2_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 2 AND Sfe Brch (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_safe_comb2_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 2 AND Sfe Brch (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_safe_comb2_adv'][2]},step=self.iter_num)

                wandb.log({'Acc Adv Cp 1,2 AND Sfe Brch (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_safe_comb_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1,2 AND Sfe Brch (USfe Cp)(Tst)': acc_comb_iter_dict['acc_safe_comb_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1,2 AND Sfe Brch (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_safe_comb_adv'][2]},step=self.iter_num)




                wandb.log({'Loss Std USfe Brch (Sfe Cp)(Tst)': loss_iter_dict['loss_unsafe_std'][0]},step=self.iter_num)
                wandb.log({'Loss Std USfe Brch (USfe Cp)(Tst)': loss_iter_dict['loss_unsafe_std'][1]},step=self.iter_num)
                wandb.log({'Loss Std USfe Brch (USfe Trg Cp)(Tst)': loss_iter_dict['loss_unsafe_std'][2]},step=self.iter_num)
                wandb.log({'Acc Next Token PCFG(Tst)': acc_pcfg_iter_dict['acc_unsafe_pcfg_std']},step=self.iter_num)

                wandb.log({'Acc Std Cp 1,2 USfe Brch (Sfe Cp)(Tst)': acc_iter_dict['acc_unsafe_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1,2 USfe Brch (USfe Cp)(Tst)': acc_iter_dict['acc_unsafe_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1,2 USfe Brch (USfe Trg Cp)(Tst)': acc_iter_dict['acc_unsafe_std'][2]},step=self.iter_num)
            
                wandb.log({'Acc Std Cp 1 USfe Brch (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_unsafe_idx1_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1 USfe Brch (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_unsafe_idx1_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1 USfe Brch (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_unsafe_idx1_std'][2]},step=self.iter_num)

                wandb.log({'Acc Std Cp 2 USfe Brch (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_unsafe_idx2_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 2 USfe Brch (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_unsafe_idx2_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 2 USfe Brch (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_unsafe_idx2_std'][2]},step=self.iter_num)
    
                wandb.log({'Acc Std Cp 1 AND USfe Brch (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_unsafe_comb1_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1 AND USfe Brch (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_unsafe_comb1_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1 AND USfe Brch (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_unsafe_comb1_std'][2]},step=self.iter_num)

                wandb.log({'Acc Std Cp 2 AND USfe Brch (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_unsafe_comb2_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 2 AND USfe Brch (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_unsafe_comb2_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 2 AND USfe Brch (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_unsafe_comb2_std'][2]},step=self.iter_num)

                wandb.log({'Acc Std Cp 1,2 AND USfe Brch (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_unsafe_comb_std'][0]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1,2 AND USfe Brch (USfe Cp)(Tst)': acc_comb_iter_dict['acc_unsafe_comb_std'][1]},step=self.iter_num)
                wandb.log({'Acc Std Cp 1,2 AND USfe Brch (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_unsafe_comb_std'][2]},step=self.iter_num)
 



                wandb.log({'Loss Adv USfe Brch (Sfe Cp)(Tst)': loss_iter_dict['loss_unsafe_adv'][0]},step=self.iter_num)
                wandb.log({'Loss Adv USfe Brch (USfe Cp)(Tst)': loss_iter_dict['loss_unsafe_adv'][1]},step=self.iter_num)
                wandb.log({'Loss Adv USfe Brch (USfe Trg Cp)(Tst)': loss_iter_dict['loss_unsafe_adv'][2]},step=self.iter_num)

                wandb.log({'Acc Adv Cp 1,2 USfe Brch (Sfe Cp)(Tst)': acc_iter_dict['acc_unsafe_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1,2 USfe Brch (USfe Cp)(Tst)': acc_iter_dict['acc_unsafe_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1,2 USfe Brch (USfe Trg Cp)(Tst)': acc_iter_dict['acc_unsafe_adv'][2]},step=self.iter_num)
            
                wandb.log({'Acc Adv Cp 1 USfe Brch (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_unsafe_idx1_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1 USfe Brch (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_unsafe_idx1_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1 USfe Brch (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_unsafe_idx1_adv'][2]},step=self.iter_num)

                wandb.log({'Acc Adv Cp 2 USfe Brch (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_unsafe_idx2_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 2 USfe Brch (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_unsafe_idx2_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 2 USfe Brch (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_unsafe_idx2_adv'][2]},step=self.iter_num)
    
                wandb.log({'Acc Adv Cp 1 AND USfe Brch (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_unsafe_comb1_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1 AND USfe Brch (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_unsafe_comb1_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1 AND USfe Brch (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_unsafe_comb1_adv'][2]},step=self.iter_num)

                wandb.log({'Acc Adv Cp 2 AND USfe Brch (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_unsafe_comb2_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 2 AND USfe Brch (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_unsafe_comb2_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 2 AND USfe Brch (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_unsafe_comb2_adv'][2]},step=self.iter_num)

                wandb.log({'Acc Adv Cp 1,2 AND USfe Brch (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_unsafe_comb_adv'][0]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1,2 AND USfe Brch (USfe Cp)(Tst)': acc_comb_iter_dict['acc_unsafe_comb_adv'][1]},step=self.iter_num)
                wandb.log({'Acc Adv Cp 1,2 AND USfe Brch (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_unsafe_comb_adv'][2]},step=self.iter_num)




                wandb.log({'Loss JB MG Txt (Sfe Cp)(Tst)': loss_iter_dict['loss_jail_mg_text'][0]},step=self.iter_num)
                wandb.log({'Loss JB MG Txt (USfe Cp)(Tst)': loss_iter_dict['loss_jail_mg_text'][1]},step=self.iter_num)
                wandb.log({'Loss JB MG Txt (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_mg_text'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Txt Cp 1,2 (Sfe Cp)(Tst)': acc_iter_dict['acc_jail_mg_text'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_mg_text'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_mg_text'][2]},step=self.iter_num)
            
                wandb.log({'Acc JB MG Txt Cp 1 (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_text'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_text'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_text'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Txt Cp 2 (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_text'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_text'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_text'][2]},step=self.iter_num)
    
                wandb.log({'Acc JB MG Txt Cp 1 AND (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_text'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_text'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_text'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Txt Cp 2 AND (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_text'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_text'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_text'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Txt Cp 1,2 AND (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_text'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_text'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Txt Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_text'][2]},step=self.iter_num)




                wandb.log({'Loss JB MG Tkns Sfe Brch (Duplicate Cap)(Tst)': loss_iter_dict['loss_jail_mg_tokens'][0]},step=self.iter_num)
                wandb.log({'Loss JB MG Tkns Sfe Brch (USfe Cp)(Tst)': loss_iter_dict['loss_jail_mg_tokens'][1]},step=self.iter_num)
                wandb.log({'Loss JB MG Tkns Sfe Brch (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_mg_tokens'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1,2 (Duplicate Cap)(Tst)': acc_iter_dict['acc_jail_mg_tokens'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_mg_tokens'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_mg_tokens'][2]},step=self.iter_num)
            
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1 (Duplicate Cap)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_tokens'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_tokens'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_tokens'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 2 (Duplicate Cap)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_tokens'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_tokens'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_tokens'][2]},step=self.iter_num)
    
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1 AND (Duplicate Cap)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_tokens'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_tokens'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_tokens'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 2 AND (Duplicate Cap)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_tokens'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_tokens'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_tokens'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1,2 AND (Duplicate Cap)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_tokens'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_tokens'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns Sfe Brch Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_tokens'][2]},step=self.iter_num)




                wandb.log({'Loss JB MG Tkns USfe Brch (Duplicate Cap)(Tst)': loss_iter_dict['loss_jail_mg_tokens2'][0]},step=self.iter_num)
                wandb.log({'Loss JB MG Tkns USfe Brch (USfe Cp)(Tst)': loss_iter_dict['loss_jail_mg_tokens2'][1]},step=self.iter_num)
                wandb.log({'Loss JB MG Tkns USfe Brch (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_mg_tokens2'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1,2 (Duplicate Cap)(Tst)': acc_iter_dict['acc_jail_mg_tokens2'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_mg_tokens2'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_mg_tokens2'][2]},step=self.iter_num)
            
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1 (Duplicate Cap)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_tokens2'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_tokens2'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_mg_tokens2'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Tkns USfe Brch Cp 2 (Duplicate Cap)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_tokens2'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_tokens2'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_mg_tokens2'][2]},step=self.iter_num)
    
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1 AND (Duplicate Cap)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_tokens2'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_tokens2'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_mg_tokens2'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Tkns USfe Brch Cp 2 AND (Duplicate Cap)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_tokens2'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_tokens2'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_mg_tokens2'][2]},step=self.iter_num)

                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1,2 AND (Duplicate Cap)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_tokens2'][0]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_tokens2'][1]},step=self.iter_num)
                wandb.log({'Acc JB MG Tkns USfe Brch Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_mg_tokens2'][2]},step=self.iter_num)



                wandb.log({'Loss JB IF (Txt) (Sfe Cp)(Tst)': loss_iter_dict['loss_jail_co'][0]},step=self.iter_num)
                wandb.log({'Loss JB IF (Txt) (USfe Cp)(Tst)': loss_iter_dict['loss_jail_co'][1]},step=self.iter_num)
                wandb.log({'Loss JB IF (Txt) (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_co'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Txt) Cp 1,2 (Sfe Cp)(Tst)': acc_iter_dict['acc_jail_co'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_co'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_co'][2]},step=self.iter_num)
            
                wandb.log({'Acc JB IF (Txt) Cp 1 (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_co'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_co'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_co'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Txt) Cp 2 (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_co'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_co'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_co'][2]},step=self.iter_num)
    
                wandb.log({'Acc JB IF (Txt) Cp 1 AND (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_co'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_co'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_co'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Txt) Cp 2 AND (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_co'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_co'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_co'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Txt) Cp 1,2 AND (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_co'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_co'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Txt) Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_co'][2]},step=self.iter_num)




                wandb.log({'Loss JB IF (Tkns) USfe Br (Sfe Cp)(Tst)': loss_iter_dict['loss_jail_if_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Loss JB IF (Tkns) USfe Br (USfe Cp)(Tst)': loss_iter_dict['loss_jail_if_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Loss JB IF (Tkns) USfe Br (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_if_tkns_unsafe'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1,2 (Sfe Cp)(Tst)': acc_iter_dict['acc_jail_if_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_if_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_if_tkns_unsafe'][2]},step=self.iter_num)
            
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1 (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_if_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_if_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_if_tkns_unsafe'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 2 (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_if_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_if_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_if_tkns_unsafe'][2]},step=self.iter_num)
    
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1 AND (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_if_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_if_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_if_tkns_unsafe'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 2 AND (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_if_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_if_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_if_tkns_unsafe'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1,2 AND (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_if_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_if_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) USfe Br Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_if_tkns_unsafe'][2]},step=self.iter_num)



                wandb.log({'Loss JB IF (Tkns) Sfe Br (Sfe Cp)(Tst)': loss_iter_dict['loss_jail_if_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Loss JB IF (Tkns) Sfe Br (USfe Cp)(Tst)': loss_iter_dict['loss_jail_if_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Loss JB IF (Tkns) Sfe Br (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_if_tkns_safe'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1,2 (Sfe Cp)(Tst)': acc_iter_dict['acc_jail_if_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_if_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_if_tkns_safe'][2]},step=self.iter_num)
            
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1 (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_if_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_if_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_if_tkns_safe'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 2 (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_if_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_if_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_if_tkns_safe'][2]},step=self.iter_num)
    
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1 AND (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_if_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_if_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_if_tkns_safe'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 2 AND (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_if_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_if_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_if_tkns_safe'][2]},step=self.iter_num)

                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1,2 AND (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_if_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_if_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB IF (Tkns) Sfe Br Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_if_tkns_safe'][2]},step=self.iter_num)





                wandb.log({'Loss JB COT (Tkns) USfe Br (Sfe Cp)(Tst)': loss_iter_dict['loss_jail_cot_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Loss JB COT (Tkns) USfe Br (USfe Cp)(Tst)': loss_iter_dict['loss_jail_cot_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Loss JB COT (Tkns) USfe Br (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_cot_tkns_unsafe'][2]},step=self.iter_num)

                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1,2 (Sfe Cp)(Tst)': acc_iter_dict['acc_jail_cot_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_cot_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_cot_tkns_unsafe'][2]},step=self.iter_num)
            
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1 (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_unsafe'][2]},step=self.iter_num)

                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 2 (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_unsafe'][2]},step=self.iter_num)
    
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1 AND (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_unsafe'][2]},step=self.iter_num)

                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 2 AND (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_unsafe'][2]},step=self.iter_num)

                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1,2 AND (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_cot_tkns_unsafe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_cot_tkns_unsafe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) USfe Br Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_cot_tkns_unsafe'][2]},step=self.iter_num)



                wandb.log({'Loss JB COT (Tkns) Sfe Br (Sfe Cp)(Tst)': loss_iter_dict['loss_jail_cot_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Loss JB COT (Tkns) Sfe Br (USfe Cp)(Tst)': loss_iter_dict['loss_jail_cot_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Loss JB COT (Tkns) Sfe Br (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_cot_tkns_safe'][2]},step=self.iter_num)

                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1,2 (Sfe Cp)(Tst)': acc_iter_dict['acc_jail_cot_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_cot_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_cot_tkns_safe'][2]},step=self.iter_num)
            
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1 (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_safe'][2]},step=self.iter_num)

                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 2 (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_safe'][2]},step=self.iter_num)
    
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1 AND (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_safe'][2]},step=self.iter_num)

                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 2 AND (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_safe'][2]},step=self.iter_num)

                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1,2 AND (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_cot_tkns_safe'][0]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_cot_tkns_safe'][1]},step=self.iter_num)
                wandb.log({'Acc JB COT (Tkns) Sfe Br Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_cot_tkns_safe'][2]},step=self.iter_num)



                if config.perform_co==True:

                    wandb.log({'Loss JB CO USfe Br (Sfe Cp)(Tst)': loss_iter_dict['loss_jail_comp_unsafe'][0]},step=self.iter_num)
                    wandb.log({'Loss JB CO USfe Br (USfe Cp)(Tst)': loss_iter_dict['loss_jail_comp_unsafe'][1]},step=self.iter_num)
                    wandb.log({'Loss JB CO USfe Br (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_comp_unsafe'][2]},step=self.iter_num)

                    wandb.log({'Acc JB CO USfe Br Cp 1,2 (Sfe Cp)(Tst)': acc_iter_dict['acc_jail_comp_unsafe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_comp_unsafe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_comp_unsafe'][2]},step=self.iter_num)
            
                    wandb.log({'Acc JB CO USfe Br Cp 1 (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_comp_unsafe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_comp_unsafe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_comp_unsafe'][2]},step=self.iter_num)

                    wandb.log({'Acc JB CO USfe Br Cp 2 (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_comp_unsafe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_comp_unsafe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_comp_unsafe'][2]},step=self.iter_num)
    
                    wandb.log({'Acc JB CO USfe Br Cp 1 AND (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_comp_unsafe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_comp_unsafe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_comp_unsafe'][2]},step=self.iter_num)

                    wandb.log({'Acc JB CO USfe Br Cp 2 AND (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_comp_unsafe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_comp_unsafe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_comp_unsafe'][2]},step=self.iter_num)

                    wandb.log({'Acc JB CO USfe Br Cp 1,2 AND (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_comp_unsafe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_comp_unsafe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO USfe Br Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_comp_unsafe'][2]},step=self.iter_num)



                    wandb.log({'Loss JB CO Sfe Br (Sfe Cp)(Tst)': loss_iter_dict['loss_jail_comp_safe'][0]},step=self.iter_num)
                    wandb.log({'Loss JB CO Sfe Br (USfe Cp)(Tst)': loss_iter_dict['loss_jail_comp_safe'][1]},step=self.iter_num)
                    wandb.log({'Loss JB CO Sfe Br (USfe Trg Cp)(Tst)': loss_iter_dict['loss_jail_comp_safe'][2]},step=self.iter_num)

                    wandb.log({'Acc JB CO Sfe Br Cp 1,2 (Sfe Cp)(Tst)': acc_iter_dict['acc_jail_comp_safe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 1,2 (USfe Cp)(Tst)': acc_iter_dict['acc_jail_comp_safe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_jail_comp_safe'][2]},step=self.iter_num)
            
                    wandb.log({'Acc JB CO Sfe Br Cp 1 (Sfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_comp_safe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 1 (USfe Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_comp_safe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_idx1_jail_comp_safe'][2]},step=self.iter_num)

                    wandb.log({'Acc JB CO Sfe Br Cp 2 (Sfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_comp_safe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 2 (USfe Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_comp_safe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_idx2_jail_comp_safe'][2]},step=self.iter_num)
    
                    wandb.log({'Acc JB CO Sfe Br Cp 1 AND (Sfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_comp_safe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 1 AND (USfe Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_comp_safe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_comb1_jail_comp_safe'][2]},step=self.iter_num)

                    wandb.log({'Acc JB CO Sfe Br Cp 2 AND (Sfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_comp_safe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 2 AND (USfe Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_comp_safe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_comb2_jail_comp_safe'][2]},step=self.iter_num)

                    wandb.log({'Acc JB CO Sfe Br Cp 1,2 AND (Sfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_comp_safe'][0]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 1,2 AND (USfe Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_comp_safe'][1]},step=self.iter_num)
                    wandb.log({'Acc JB CO Sfe Br Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_comb_jail_comp_safe'][2]},step=self.iter_num)



                if config.univ_adv==True:

                    wandb.log({'Loss Univ Adv USfe Br (USfe Trg Cp)(Tst)': loss_iter_dict['loss_unsafe_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv USfe Br Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_unsafe_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv USfe Br Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_unsafe_idx1_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv USfe Br Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_unsafe_idx2_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv USfe Br Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_unsafe_comb1_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv USfe Br Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_unsafe_comb2_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv USfe Br Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_unsafe_comb_univ_adv'][2]},step=self.iter_num)



      
                    wandb.log({'Loss Univ Adv Sfe Br (USfe Trg Cp)(Tst)': loss_iter_dict['loss_safe_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv Sfe Br Cp 1,2 (USfe Trg Cp)(Tst)': acc_iter_dict['acc_safe_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv Sfe Br Cp 1 (USfe Trg Cp)(Tst)': acc_idx1_iter_dict['acc_safe_idx1_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv Sfe Br Cp 2 (USfe Trg Cp)(Tst)': acc_idx2_iter_dict['acc_safe_idx2_univ_adv'][2]},step=self.iter_num)

                    wandb.log({'Acc Univ Adv Sfe Br Cp 1 AND (USfe Trg Cp)(Tst)': acc_comb1_iter_dict['acc_safe_comb1_univ_adv'][2]},step=self.iter_num)
      
                    wandb.log({'Acc Univ Adv Sfe Br Cp 2 AND (USfe Trg Cp)(Tst)': acc_comb2_iter_dict['acc_safe_comb2_univ_adv'][2]},step=self.iter_num)
        
                    wandb.log({'Acc Univ Adv Sfe Br Cp 1,2 AND (USfe Trg Cp)(Tst)': acc_comb_iter_dict['acc_safe_comb_univ_adv'][2]},step=self.iter_num)

                print("Testing Done")


            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break


    def adv_attack(self, model, x_safe_cap, x_unsafe_cap, x_unsafe_cap_target, y_safe_cap, y_unsafe_cap, y_unsafe_cap_target, mask, start_idx, end_idx, idx, idx1, idx2, idx3,univ_adv=False,steps=0):
        if steps==0:
            steps=self.config.adv_attack_iters

        model.eval()
        lst_x_adv_imgs = []
        lst_y_adv_imgs = []
        if univ_adv==False:
            idx_vals = [0,1,2]
        elif univ_adv==True:
            idx_vals = [2]
        for idx in idx_vals:
            if idx==0:
                x_img = x_safe_cap
                y_img = y_safe_cap
                adv_attack_sign = 1
            
            elif idx==1:
                x_img = x_unsafe_cap
                y_img = y_unsafe_cap
                adv_attack_sign = 1
            
            elif idx==2:
                x_img = x_unsafe_cap_target
                y_img = y_unsafe_cap_target
                adv_attack_sign = -1
            
            arr_x = x_img.detach().cpu().numpy().copy()
            arr_y = y_img.detach().cpu().numpy().copy()
            num_new_pos = self.config.threat_count_adv
            adv_tokens = []

            lst_x = []
            lst_mask = []
            lst_y = []
            lst_mask_y = []

            for i in range(num_new_pos):
                adv_tokens.append(0)
            for i in range(arr_x.shape[0]):
                arr_slice = arr_x[i].tolist()
                arr_slice_y = arr_y[i].tolist()


                if self.config.threat_pos_adv==-1:
                    arr_mod = arr_slice[:start_idx[i]]+ adv_tokens + arr_slice[start_idx[i]:end_idx[i][0]] + adv_tokens + arr_slice[end_idx[i][0]:end_idx[i][1]] + adv_tokens + arr_slice[end_idx[i][1]:end_idx[i][1]+2]
                    arr_mod = arr_mod + (self.config.max_window_possible-len(arr_mod))*[self.config.tokenizer[self.config.pad_token]]
                    arr_mask = len(arr_slice[:start_idx[i]])*[0] + len(adv_tokens)*[1] + len(arr_slice[start_idx[i]:end_idx[i][0]])*[0] + len(adv_tokens)*[0] + len(arr_slice[end_idx[i][0]:end_idx[i][1]])*[0] + len(adv_tokens)*[0] +  len(arr_slice[end_idx[i][1]:end_idx[i][1]+2])*[0]
                    arr_mask = arr_mask + (self.config.max_window_possible-len(arr_mask))*[0]
                
                    arr_mod_y = arr_slice_y[:start_idx[i]-1]+ len(adv_tokens)*[self.config.tokenizer[self.config.pad_token]] + arr_slice_y[start_idx[i]-1:end_idx[i][0]-1] + len(adv_tokens)*[self.config.tokenizer[self.config.pad_token]] + arr_slice_y[end_idx[i][0]-1:end_idx[i][1]-1] + len(adv_tokens)*[self.config.tokenizer[self.config.pad_token]] + arr_slice_y[end_idx[i][1]-1:end_idx[i][1]+1]
                    arr_mod_y = arr_mod_y + (self.config.max_window_possible-len(arr_mod_y))*[self.config.tokenizer[self.config.pad_token]]
                    arr_mask_y = len(arr_slice[:start_idx[i]-1])*[0] + len(adv_tokens)*[0] + [0] + len(arr_slice[start_idx[i]:end_idx[i][0]-1])*[1] + len(adv_tokens)*[0] + [0] + len(arr_slice[end_idx[i][0]:end_idx[i][1]-1])*[1] + len(adv_tokens)*[0] + [0] + len(arr_slice[end_idx[i][1]:end_idx[i][1]+1])*[0]
                    arr_mask_y = arr_mask_y + (self.config.max_window_possible-len(arr_mask_y))*[0]
                
                #if idx==2:
                #    print("arr_slice",arr_slice)
                #    print("arr_mod", arr_mod)
                #    print("arr_mod_y", arr_mod_y)
                #    print("arr_mask", arr_mask)
                #    print("arr_mask_y", arr_mask_y)
                #    #a+=1



                # else:
                #     arr_mod = arr_slice[:start_idx[i]-num_new_pos]+ adv_tokens + arr_slice[start_idx[i]-num_new_pos:end_idx[i][0]-num_new_pos] + adv_tokens + arr_slice[end_idx[i][0]-num_new_pos:end_idx[i][1]-num_new_pos] + adv_tokens + arr_slice[end_idx[i][1]-num_new_pos:end_idx[i][1]]
                #     arr_mod = arr_mod + (self.config.max_window_possible-len(arr_mod))*[self.config.tokenizer[self.config.pad_token]]
                #     arr_mask = len(arr_slice[:start_idx[i]-num_new_pos])*[0] + len(adv_tokens)*[1] + len(arr_slice[start_idx[i]-num_new_pos:end_idx[i][0]-num_new_pos])*[0] + len(adv_tokens)*[0] + len(arr_slice[end_idx[i][0]-num_new_pos:end_idx[i][1]-num_new_pos])*[0] + len(adv_tokens)*[0] + len(arr_slice[end_idx[i][1]-num_new_pos:end_idx[i][1]])*[0]
                #     arr_mask = arr_mask + (self.config.max_window_possible-len(arr_mask))*[0]
                
                #     arr_mod_y = arr_slice_y[:start_idx[i]-num_new_pos]+ len(adv_tokens)*[self.config.tokenizer[self.config.pad_token]] + arr_slice_y[start_idx[i]-num_new_pos:end_idx[i][0]-num_new_pos] + len(adv_tokens)*[self.config.tokenizer[self.config.pad_token]] + arr_slice_y[-num_new_pos+end_idx[i][0]:end_idx[i][1]-num_new_pos] + len(adv_tokens)*[self.config.tokenizer[self.config.pad_token]]
                #     arr_mod_y = arr_mod_y + (self.config.max_window_possible-len(arr_mod_y))*[self.config.tokenizer[self.config.pad_token]]
                #     arr_mask_y = len(arr_slice[:start_idx[i]-num_new_pos])*[0] + len(adv_tokens)*[0] + len(arr_slice[start_idx[i]-num_new_pos:start_idx[i]])*[0] + len(arr_slice[start_idx[i]:-num_new_pos+end_idx[i][0]])*[1] + len(adv_tokens)*[0] + len(arr_slice[-num_new_pos+end_idx[i][0]:-num_new_pos+end_idx[i][1]])*[1] + len(adv_tokens)*[0] + len(arr_slice[end_idx[i][1]-num_new_pos:end_idx[i][1]])*[0] 
                #     arr_mask_y = arr_mask_y + (self.config.max_window_possible-len(arr_mask_y))*[0]


                #print("Arr mod", len(arr_mod_y))
            
                #print("X", arr_slice)
                #print("Y", arr_slice_y)
                #print("Mod X", arr_mod)
                #print("Mod Y", arr_mod_y)
                #print("Mask X", arr_mask)
                #print("Mask Y", arr_mask_y)
                #a+=1

                lst_x.append(arr_mod)
                lst_mask.append(arr_mask)
                lst_y.append(arr_mod_y)
                lst_mask_y.append(arr_mask_y)

            x = torch.LongTensor(np.array(lst_x)).cuda()
            y = torch.LongTensor(np.array(lst_y)).cuda()
            tok_emb = model(x, y, mask, attack='emb').detach()
            mask_x = torch.LongTensor(np.array(lst_mask)).cuda().unsqueeze(-1)
            mask_y = torch.LongTensor(np.array(lst_mask_y)).cuda().unsqueeze(-1)

        
            allowed_norm, min_norm, max_norm = self.get_norm(tok_emb, norm=self.config.adv_attack_norm)
            #print("Allowed norm1", allowed_norm)

            #print("Shape of mask", mask_x.shape)
            #print("Shape of token emb", tok_emb.shape)
            x_natural = torch.mul(1-mask_x,tok_emb.cuda().detach())
            if self.config.adv_attack_norm=='fro': 
                delta = torch.mul(mask_x, 0.01*torch.randn(tok_emb.shape).cuda().detach())
            else:
                delta = torch.mul(mask_x, 0.001*torch.randn(tok_emb.shape).cuda().detach())
    
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=allowed_norm / (steps) * 2)

            for _ in range(steps):            
                # optimize
                x_adv = x_natural + delta
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    out, loss = model(x_adv, y, mask, attack='emb_in', emb=x_adv)
                    attack_loss = -adv_attack_sign*loss
                attack_loss.backward()
                #if idx==2:
                #    print("Iteration loss", attack_loss)

                # renorming gradient
                for i in range(delta.shape[0]):
                    for j in range(num_new_pos):
                        if self.config.adv_attack_norm=='fro':
                            if self.config.threat_pos_adv==-1:
                                grad_norms = delta.grad[i][start_idx[i]-1+j].norm(p=2)
                                delta.grad[i][start_idx[i]-1+j].div_(grad_norms)
                            else:
                                grad_norms = delta.grad[i][start_idx[i]-1+j].norm(p=2)
                                delta.grad[i][start_idx[i]-num_new_pos+j].div_(grad_norms)

                            if (grad_norms == 0).any():
                                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                        elif self.config.adv_attack_norm=='inf':
                            if self.config.threat_pos_adv==-1:
                                delta[i][start_idx[i]-1+j].grad.clamp_(min_norm,allowed_norm)
                            else:
                                delta[i][start_idx[i]-num_new_pos+j].grad.clamp_(min_norm,allowed_norm)
                            #delta.grad[i][start_idx[i]+j].div_(grad_norms)

                # avoid nan or inf if gradient is 0
                optimizer_delta.step()

                # projection
                delta.data = torch.mul(mask_x, delta.data)
                #delta.data.add_(x_natural)
                #delta.data.clamp_(0, 1).sub_(x_natural)
                if univ_adv==True:
                    avg_grad = []
                    for j in range(num_new_pos):
                        avg_grad.append(0)
                    for i in range(delta.shape[0]):
                        for j in range(num_new_pos):
                            if self.config.adv_attack_norm=='fro':
                                #print(delta[i][start_idx[i]+j].unsqueeze(-1).shape)
                                #a = delta[i][start_idx[i]+j].clone()
                                #print("Norm value", torch.norm(delta[i][start_idx[i]-1+j]))
                                #print("Max norm", allowed_norm)
                                #print(delta.shape)
                                #a+=1
                                if i==0:
                                    avg_grad[j] = delta[i][start_idx[i]-1+j].data.unsqueeze(-1)
                                else:
                                    avg_grad[j]+=delta[i][start_idx[i]-1+j].data.unsqueeze(-1)

                    for i in range(delta.shape[0]):
                        for j in range(num_new_pos):
                            delta[i][start_idx[i]-1+j].data = torch.renorm(avg_grad[j]/delta.shape[0], p=2, dim=0, maxnorm=allowed_norm)[:,0]

                else:
                    for i in range(delta.shape[0]):
                        for j in range(num_new_pos):
                            if self.config.adv_attack_norm=='fro':
                                #print(delta[i][start_idx[i]+j].unsqueeze(-1).shape)
                                #a = delta[i][start_idx[i]+j].clone()
                                #print("Norm value", torch.norm(delta[i][start_idx[i]-1+j]))
                                #print("Max norm", allowed_norm)
                                #print(delta.shape)
                                #a+=1
                                delta[i][start_idx[i]-1+j].data.unsqueeze(-1).renorm_(p=2, dim=0, maxnorm=allowed_norm)
                                #print(a-delta[i][start_idx[i]+j])
                            elif self.config.adv_attack_norm=='inf':
                                delta[i][start_idx[i]-1+j].data.clamp_(min_norm,allowed_norm)

            x_safe_cap_adv = Variable(x_natural + torch.mul(mask_x, delta), requires_grad=False)

    
            lst_x_adv_imgs.append(x_safe_cap_adv)
            lst_y_adv_imgs.append(y)

        if univ_adv==False:
            return lst_x_adv_imgs[0], lst_x_adv_imgs[1], lst_x_adv_imgs[2], lst_y_adv_imgs[0], lst_y_adv_imgs[1], lst_y_adv_imgs[2] 
        else:
            return lst_x_adv_imgs[0], lst_y_adv_imgs[0] 




        
    def get_norm(self, tok_emb, norm='fro'):
        counter=0
        min_norm=99999
        max_norm=0
        norm_sum = 0
        for i in range(tok_emb.shape[0]):
            for j in range(tok_emb.shape[1]):
                norm_value = torch.norm(tok_emb[i][j],p=norm)
                norm_sum+=norm_value
                if norm_value<min_norm:
                    min_norm = norm_value
                if norm_value>max_norm:
                    max_norm = norm_value
                counter+=1
        
        return norm_sum/counter, min_norm, max_norm



    def get_log(self,logits, y, loss, start_idx, end_idx, idx, idx1, idx2, idx3):
        acc_comb1=0
        acc_comb2=0
        acc_comb2_other = 0
        counter_idx2_other = 0
        acc = 0
        acc_comb=0
        acc_idx1 = 0
        acc_idx2 = 0 
        acc_pcfg = 0
        counter_acc = 0
        acc_idx2_other = 0
        counter_comb1 = 0
        counter_comb2 = 0
        counter_comb_all = 0
        #print("Y shape", y.shape)
        #print("Logits shape", logits.shape)
        for i in range(y.shape[0]):
            counter_acc+=1
            if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token] and y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                acc_temp = (logits[i][start_idx[i]:end_idx[i][-1]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][-1]])
                acc+=torch.sum(acc_temp/acc_temp.shape[0]).item()
                
                if torch.sum(acc_temp) == (end_idx[i][-1]-start_idx[i]):
                    acc_comb+=1
                counter_comb_all+=1

            acc_temp_pcfg= (logits[i][1+self.config.num_cap:start_idx[i]].argmax(dim=-1)==y[i][1+self.config.num_cap:start_idx[i]])
            acc_pcfg+=torch.sum(acc_temp_pcfg/acc_temp_pcfg.shape[0]).item()

            if y[i][start_idx[i]]!=self.config.tokenizer[self.config.pad_token]:
                acc_idx1_temp = (logits[i][start_idx[i]:end_idx[i][0]].argmax(dim=-1)==y[i][start_idx[i]:end_idx[i][0]])
                acc_idx1+=torch.sum(acc_idx1_temp/acc_idx1_temp.shape[0]).item()
                if torch.sum(acc_idx1_temp)==(end_idx[i][0]-start_idx[i]):
                    acc_comb1+=1
                counter_comb1+=1

            if y[i][end_idx[i][0]]!=self.config.tokenizer[self.config.pad_token]:
                acc_idx2_temp = (logits[i][end_idx[i][0]:end_idx[i][1]].argmax(dim=-1)==y[i][end_idx[i][0]:end_idx[i][1]])
                acc_idx2+=torch.sum(acc_idx2_temp/acc_idx2_temp.shape[0]).item()
                if torch.sum(acc_idx2_temp)==(end_idx[i][1]-end_idx[i][0]):
                    acc_comb2+=1
                counter_comb2+=1

                            
        try:
            size_lst = loss.shape[0]
            loss_return = torch.sum(loss)/loss.shape[0]
        except:
            loss_return = loss
        
        if counter_comb_all==0:
            counter_comb_all=1

        if counter_acc==0:
            counter_acc=1

        if counter_comb1==0:
            counter_comb1=1

        if counter_comb2==0:
            counter_comb2=1


        acc_iter = acc/counter_comb_all
        acc_iter_pcfg = acc_pcfg/counter_acc
        acc_idx1_iter = acc_idx1/counter_comb1
        acc_idx2_iter = acc_idx2/counter_comb2
        acc_comb1_iter = acc_comb1/counter_comb1
        acc_comb2_iter = acc_comb2/counter_comb2
        acc_comb_iter = acc_comb/counter_comb_all
        loss_iter=loss_return

        return loss_iter, acc_iter,acc_iter_pcfg, acc_idx1_iter, acc_idx2_iter,  acc_comb1_iter, acc_comb2_iter, acc_comb_iter



    def evaluate_pretrain(self, model, test_loader_safe, test_loader_unsafe, test_loader_intermediate, test_loader_all, test_loader_duplicate, test_loader_id_mg, test_loader_ood_mg, test_loader_id_jailbreak_comp, test_loader_safe_jailbreak_comp, test_loader_id_jailbreak_direct, test_loader_safe_jailbreak_direct,test_loader_duplicate2, max_iterations=0,attack_lst=[]):
        model.eval()
        net_acc = 0.0
        net_acc_token = 0.0
        loss = 0.0
        loss_total_AT = 0.0
        acc_iter = 0
        acc_idx1_iter = 0
        acc_idx2_iter = 0
        acc_comb1_iter = 0
        acc_comb2_iter = 0
        acc_comb_iter = 0
        iter_counter = 0
        loss_iter = 0
        loss_iter_dict={}
        acc_iter_dict={}
        acc_idx1_iter_dict={}
        acc_idx2_iter_dict={}
        acc_comb1_iter_dict={}
        acc_comb2_iter_dict={}
        acc_comb_iter_dict={}
        acc_pcfg_iter_dict={}


        loss_iter_dict['loss_safe_std']=[0, 0, 0]
        acc_iter_dict['acc_safe_std']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_safe_pcfg_std']=[0, 0, 0]
        acc_idx1_iter_dict['acc_safe_idx1_std']=[0, 0, 0]
        acc_idx2_iter_dict['acc_safe_idx2_std']=[0, 0, 0]
        acc_comb1_iter_dict['acc_safe_comb1_std']=[0, 0, 0]
        acc_comb2_iter_dict['acc_safe_comb2_std']=[0, 0, 0]
        acc_comb_iter_dict['acc_safe_comb_std']=[0, 0, 0]

        loss_iter_dict['loss_safe_adv']=[0, 0, 0]
        acc_iter_dict['acc_safe_adv']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_safe_pcfg_adv']=[0, 0, 0]
        acc_idx1_iter_dict['acc_safe_idx1_adv']=[0, 0, 0]
        acc_idx2_iter_dict['acc_safe_idx2_adv']=[0, 0, 0]
        acc_comb1_iter_dict['acc_safe_comb1_adv']=[0, 0, 0]
        acc_comb2_iter_dict['acc_safe_comb2_adv']=[0, 0, 0]
        acc_comb_iter_dict['acc_safe_comb_adv']=[0, 0, 0]        

        loss_iter_dict['loss_unsafe_std']=[0, 0, 0]
        acc_iter_dict['acc_unsafe_std']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_unsafe_pcfg_std']=[0, 0, 0]
        acc_idx1_iter_dict['acc_unsafe_idx1_std']=[0, 0, 0]
        acc_idx2_iter_dict['acc_unsafe_idx2_std']=[0, 0, 0]
        acc_comb1_iter_dict['acc_unsafe_comb1_std']=[0, 0, 0]
        acc_comb2_iter_dict['acc_unsafe_comb2_std']=[0, 0, 0]
        acc_comb_iter_dict['acc_unsafe_comb_std']=[0, 0, 0]

        loss_iter_dict['loss_unsafe_adv']=[0, 0, 0]
        acc_iter_dict['acc_unsafe_adv']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_unsafe_pcfg_adv']=[0, 0, 0]
        acc_idx1_iter_dict['acc_unsafe_idx1_adv']=[0, 0, 0]
        acc_idx2_iter_dict['acc_unsafe_idx2_adv']=[0, 0, 0]
        acc_comb1_iter_dict['acc_unsafe_comb1_adv']=[0, 0, 0]
        acc_comb2_iter_dict['acc_unsafe_comb2_adv']=[0, 0, 0]
        acc_comb_iter_dict['acc_unsafe_comb_adv']=[0, 0, 0]


        loss_iter_dict['loss_jail_mg_tokens2']=[0, 0, 0]
        acc_iter_dict['acc_jail_mg_tokens2']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_pcfg_jail_mg_tokens2']=[0, 0, 0]
        acc_idx1_iter_dict['acc_idx1_jail_mg_tokens2']=[0, 0, 0]
        acc_idx2_iter_dict['acc_idx2_jail_mg_tokens2']=[0, 0, 0]
        acc_comb1_iter_dict['acc_comb1_jail_mg_tokens2']=[0, 0, 0]
        acc_comb2_iter_dict['acc_comb2_jail_mg_tokens2']=[0, 0, 0]
        acc_comb_iter_dict['acc_comb_jail_mg_tokens2']=[0, 0, 0]

        loss_iter_dict['loss_safe_univ_adv']=[0, 0, 0]
        acc_iter_dict['acc_safe_univ_adv']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_safe_pcfg_univ_adv']=[0, 0, 0]
        acc_idx1_iter_dict['acc_safe_idx1_univ_adv']=[0, 0, 0]
        acc_idx2_iter_dict['acc_safe_idx2_univ_adv']=[0, 0, 0]
        acc_comb1_iter_dict['acc_safe_comb1_univ_adv']=[0, 0, 0]
        acc_comb2_iter_dict['acc_safe_comb2_univ_adv']=[0, 0, 0]
        acc_comb_iter_dict['acc_safe_comb_univ_adv']=[0, 0, 0]        


        loss_iter_dict['loss_unsafe_univ_adv']=[0, 0, 0]
        acc_iter_dict['acc_unsafe_univ_adv']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_unsafe_pcfg_univ_adv']=[0, 0, 0]
        acc_idx1_iter_dict['acc_unsafe_idx1_univ_adv']=[0, 0, 0]
        acc_idx2_iter_dict['acc_unsafe_idx2_univ_adv']=[0, 0, 0]
        acc_comb1_iter_dict['acc_unsafe_comb1_univ_adv']=[0, 0, 0]
        acc_comb2_iter_dict['acc_unsafe_comb2_univ_adv']=[0, 0, 0]
        acc_comb_iter_dict['acc_unsafe_comb_univ_adv']=[0, 0, 0]


        loss_iter_dict['loss_jail_mg_text']=[0, 0, 0]
        acc_iter_dict['acc_jail_mg_text']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_pcfg_jail_mg_text']=[0, 0, 0]
        acc_idx1_iter_dict['acc_idx1_jail_mg_text']=[0, 0, 0]
        acc_idx2_iter_dict['acc_idx2_jail_mg_text']=[0, 0, 0]
        acc_comb1_iter_dict['acc_comb1_jail_mg_text']=[0, 0, 0]
        acc_comb2_iter_dict['acc_comb2_jail_mg_text']=[0, 0, 0]
        acc_comb_iter_dict['acc_comb_jail_mg_text']=[0, 0, 0]

        loss_iter_dict['loss_jail_mg_tokens']=[0, 0, 0]
        acc_iter_dict['acc_jail_mg_tokens']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_pcfg_jail_mg_tokens']=[0, 0, 0]
        acc_idx1_iter_dict['acc_idx1_jail_mg_tokens']=[0, 0, 0]
        acc_idx2_iter_dict['acc_idx2_jail_mg_tokens']=[0, 0, 0]
        acc_comb1_iter_dict['acc_comb1_jail_mg_tokens']=[0, 0, 0]
        acc_comb2_iter_dict['acc_comb2_jail_mg_tokens']=[0, 0, 0]
        acc_comb_iter_dict['acc_comb_jail_mg_tokens']=[0, 0, 0]

        loss_iter_dict['loss_jail_co']=[0, 0, 0]
        acc_iter_dict['acc_jail_co']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_pcfg_jail_co']=[0, 0, 0]
        acc_idx1_iter_dict['acc_idx1_jail_co']=[0, 0, 0]
        acc_idx2_iter_dict['acc_idx2_jail_co']=[0, 0, 0]
        acc_comb1_iter_dict['acc_comb1_jail_co']=[0, 0, 0]
        acc_comb2_iter_dict['acc_comb2_jail_co']=[0, 0, 0]
        acc_comb_iter_dict['acc_comb_jail_co']=[0, 0, 0]

        loss_iter_dict['loss_jail_if_tkns_unsafe']=[0, 0, 0]
        acc_iter_dict['acc_jail_if_tkns_unsafe']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_pcfg_jail_if_tkns_unsafe']=[0, 0, 0]
        acc_idx1_iter_dict['acc_idx1_jail_if_tkns_unsafe']=[0, 0, 0]
        acc_idx2_iter_dict['acc_idx2_jail_if_tkns_unsafe']=[0, 0, 0]
        acc_comb1_iter_dict['acc_comb1_jail_if_tkns_unsafe']=[0, 0, 0]
        acc_comb2_iter_dict['acc_comb2_jail_if_tkns_unsafe']=[0, 0, 0]
        acc_comb_iter_dict['acc_comb_jail_if_tkns_unsafe']=[0, 0, 0]


        loss_iter_dict['loss_jail_if_tkns_safe']=[0, 0, 0]
        acc_iter_dict['acc_jail_if_tkns_safe']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_pcfg_jail_if_tkns_safe']=[0, 0, 0]
        acc_idx1_iter_dict['acc_idx1_jail_if_tkns_safe']=[0, 0, 0]
        acc_idx2_iter_dict['acc_idx2_jail_if_tkns_safe']=[0, 0, 0]
        acc_comb1_iter_dict['acc_comb1_jail_if_tkns_safe']=[0, 0, 0]
        acc_comb2_iter_dict['acc_comb2_jail_if_tkns_safe']=[0, 0, 0]
        acc_comb_iter_dict['acc_comb_jail_if_tkns_safe']=[0, 0, 0]

        loss_iter_dict['loss_jail_cot_tkns_unsafe']=[0, 0, 0]
        acc_iter_dict['acc_jail_cot_tkns_unsafe']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_pcfg_jail_cot_tkns_unsafe']=[0, 0, 0]
        acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_unsafe']=[0, 0, 0]
        acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_unsafe']=[0, 0, 0]
        acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_unsafe']=[0, 0, 0]
        acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_unsafe']=[0, 0, 0]
        acc_comb_iter_dict['acc_comb_jail_cot_tkns_unsafe']=[0, 0, 0]


        loss_iter_dict['loss_jail_cot_tkns_safe']=[0, 0, 0]
        acc_iter_dict['acc_jail_cot_tkns_safe']=[0, 0, 0]
        acc_pcfg_iter_dict['acc_pcfg_jail_cot_tkns_safe']=[0, 0, 0]
        acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_safe']=[0, 0, 0]
        acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_safe']=[0, 0, 0]
        acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_safe']=[0, 0, 0]
        acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_safe']=[0, 0, 0]
        acc_comb_iter_dict['acc_comb_jail_cot_tkns_safe']=[0, 0, 0]



        iter_counter=0

        for batch in test_loader_safe:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)

            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            ########## adversarial attack #####################
            
            x_safe_cap_adv, x_unsafe_cap_adv, x_unsafe_cap_target_adv, y_safe_cap_adv, y_unsafe_cap_adv, y_unsafe_cap_target_adv = self.adv_attack(model, x_safe_cap, x_unsafe_cap, x_unsafe_cap_target, y_safe_cap, y_unsafe_cap, y_unsafe_cap_target,mask, start_idx, end_idx, idx, idx1, idx2, idx3)
            out_safe, loss_safe = model(x_safe_cap_adv, y_safe_cap_adv, mask, attack='emb_in', emb=x_safe_cap_adv)
            out_unsafe, loss_unsafe = model(x_unsafe_cap_adv, y_unsafe_cap_adv, mask, attack='emb_in', emb=x_unsafe_cap_adv)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target_adv, y_unsafe_cap_target_adv, mask, attack='emb_in', emb=x_unsafe_cap_target_adv)

            start_idx = start_idx + self.config.threat_count_adv
            end_idx[0] = end_idx[0] + 2*self.config.threat_count_adv
            end_idx[1] = end_idx[1] + 3*self.config.threat_count_adv


            loss_iter_adv_safe, acc_iter_adv_safe, acc_iter_adv_pcfg_safe, acc_idx1_iter_adv_safe, acc_idx2_iter_adv_safe, acc_comb1_iter_adv_safe, acc_comb2_iter_adv_safe, acc_comb_iter_adv_safe = self.get_log(out_safe, y_safe_cap_adv, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_adv_unsafe, acc_iter_adv_unsafe, acc_iter_adv_pcfg_unsafe, acc_idx1_iter_adv_unsafe, acc_idx2_iter_adv_unsafe, acc_comb1_iter_adv_unsafe, acc_comb2_iter_adv_unsafe, acc_comb_iter_adv_unsafe = self.get_log(out_unsafe, y_unsafe_cap_adv, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_adv_unsafe_target, acc_iter_adv_unsafe_target, acc_iter_adv_pcfg_unsafe_target, acc_idx1_iter_adv_unsafe_target, acc_idx2_iter_adv_unsafe_target, acc_comb1_iter_adv_unsafe_target, acc_comb2_iter_adv_unsafe_target, acc_comb_iter_adv_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target_adv, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)


            ########## universal adversarial attack #####################
            
            x_unsafe_cap_target_adv, y_unsafe_cap_target_adv = self.adv_attack(model, x_safe_cap, x_unsafe_cap, x_unsafe_cap_target, y_safe_cap, y_unsafe_cap, y_unsafe_cap_target,mask, start_idx, end_idx, idx, idx1, idx2, idx3, univ_adv=True, steps=100)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target_adv, y_unsafe_cap_target_adv, mask, attack='emb_in', emb=x_unsafe_cap_target_adv)

            start_idx = start_idx + self.config.threat_count_adv
            end_idx[0] = end_idx[0] + 2*self.config.threat_count_adv
            end_idx[1] = end_idx[1] + 3*self.config.threat_count_adv

            loss_iter_adv_unsafe_target_univ, acc_iter_adv_unsafe_target_univ, acc_iter_adv_pcfg_unsafe_target_univ, acc_idx1_iter_adv_unsafe_target_univ, acc_idx2_iter_adv_unsafe_target_univ, acc_comb1_iter_adv_unsafe_target_univ, acc_comb2_iter_adv_unsafe_target_univ, acc_comb_iter_adv_unsafe_target_univ = self.get_log(out_unsafe_target, y_unsafe_cap_target_adv, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_safe_std'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_safe_std'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_safe_pcfg_std'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_safe_idx1_std'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_safe_idx2_std'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_safe_comb1_std'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_safe_comb2_std'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_safe_comb_std'][0]+=acc_comb_iter_safe


            loss_iter_dict['loss_safe_std'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_safe_std'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_safe_idx1_std'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_safe_idx2_std'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_safe_comb1_std'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_safe_comb2_std'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_safe_comb_std'][1]+=acc_comb_iter_unsafe



            loss_iter_dict['loss_safe_std'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_safe_std'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_safe_idx1_std'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_safe_idx2_std'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_safe_comb1_std'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_safe_comb2_std'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_safe_comb_std'][2]+=acc_comb_iter_unsafe_target


            loss_iter_dict['loss_safe_adv'][0]+=loss_iter_adv_safe.item()
            acc_iter_dict['acc_safe_adv'][0]+=acc_iter_adv_safe
            acc_pcfg_iter_dict['acc_safe_pcfg_adv'][0]+=acc_iter_adv_pcfg_safe
            acc_idx1_iter_dict['acc_safe_idx1_adv'][0]+=acc_idx1_iter_adv_safe
            acc_idx2_iter_dict['acc_safe_idx2_adv'][0]+=acc_idx2_iter_adv_safe
            acc_comb1_iter_dict['acc_safe_comb1_adv'][0]+=acc_comb1_iter_adv_safe
            acc_comb2_iter_dict['acc_safe_comb2_adv'][0]+=acc_comb2_iter_adv_safe
            acc_comb_iter_dict['acc_safe_comb_adv'][0]+=acc_comb_iter_adv_safe


            loss_iter_dict['loss_safe_adv'][1]+=loss_iter_adv_unsafe.item()
            acc_iter_dict['acc_safe_adv'][1]+=acc_iter_adv_unsafe
            acc_pcfg_iter_dict['acc_safe_pcfg_adv'][1]+=acc_iter_adv_pcfg_unsafe
            acc_idx1_iter_dict['acc_safe_idx1_adv'][1]+=acc_idx1_iter_adv_unsafe
            acc_idx2_iter_dict['acc_safe_idx2_adv'][1]+=acc_idx2_iter_adv_unsafe
            acc_comb1_iter_dict['acc_safe_comb1_adv'][1]+=acc_comb1_iter_adv_unsafe
            acc_comb2_iter_dict['acc_safe_comb2_adv'][1]+=acc_comb2_iter_adv_unsafe
            acc_comb_iter_dict['acc_safe_comb_adv'][1]+=acc_comb_iter_adv_unsafe



            loss_iter_dict['loss_safe_adv'][2]+=loss_iter_adv_unsafe_target.item()
            acc_iter_dict['acc_safe_adv'][2]+=acc_iter_adv_unsafe_target
            acc_pcfg_iter_dict['acc_safe_pcfg_adv'][2]+=acc_iter_adv_pcfg_unsafe_target
            acc_idx1_iter_dict['acc_safe_idx1_adv'][2]+=acc_idx1_iter_adv_unsafe_target
            acc_idx2_iter_dict['acc_safe_idx2_adv'][2]+=acc_idx2_iter_adv_unsafe_target
            acc_comb1_iter_dict['acc_safe_comb1_adv'][2]+=acc_comb1_iter_adv_unsafe_target
            acc_comb2_iter_dict['acc_safe_comb2_adv'][2]+=acc_comb2_iter_adv_unsafe_target
            acc_comb_iter_dict['acc_safe_comb_adv'][2]+=acc_comb_iter_adv_unsafe_target




            loss_iter_dict['loss_safe_univ_adv'][2]+=loss_iter_adv_unsafe_target_univ.item()
            acc_iter_dict['acc_safe_univ_adv'][2]+=acc_iter_adv_unsafe_target_univ
            acc_pcfg_iter_dict['acc_safe_pcfg_univ_adv'][2]+=acc_iter_adv_pcfg_unsafe_target_univ
            acc_idx1_iter_dict['acc_safe_idx1_univ_adv'][2]+=acc_idx1_iter_adv_unsafe_target_univ
            acc_idx2_iter_dict['acc_safe_idx2_univ_adv'][2]+=acc_idx2_iter_adv_unsafe_target_univ
            acc_comb1_iter_dict['acc_safe_comb1_univ_adv'][2]+=acc_comb1_iter_adv_unsafe_target_univ
            acc_comb2_iter_dict['acc_safe_comb2_univ_adv'][2]+=acc_comb2_iter_adv_unsafe_target_univ
            acc_comb_iter_dict['acc_safe_comb_univ_adv'][2]+=acc_comb_iter_adv_unsafe_target_univ


            if iter_counter>=max_iterations:
                break

        iter_counter=0

        for batch in test_loader_unsafe:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)

            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            ########## adversarial attack #####################  
            x_safe_cap_adv, x_unsafe_cap_adv, x_unsafe_cap_target_adv, y_safe_cap_adv, y_unsafe_cap_adv, y_unsafe_cap_target_adv = self.adv_attack(model, x_safe_cap, x_unsafe_cap, x_unsafe_cap_target, y_safe_cap, y_unsafe_cap, y_unsafe_cap_target,mask, start_idx, end_idx, idx, idx1, idx2, idx3)

            out_safe, loss_safe = model(x_safe_cap_adv, y_safe_cap_adv, mask, attack='emb_in', emb=x_safe_cap_adv)
            out_unsafe, loss_unsafe = model(x_unsafe_cap_adv, y_unsafe_cap_adv, mask, attack='emb_in', emb=x_unsafe_cap_adv)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target_adv, y_unsafe_cap_target_adv, mask, attack='emb_in', emb=x_unsafe_cap_target_adv)

            start_idx = start_idx + self.config.threat_count_adv
            end_idx[0] = end_idx[0] + 2*self.config.threat_count_adv
            end_idx[1] = end_idx[1] + 3*self.config.threat_count_adv


            loss_iter_adv_safe, acc_iter_adv_safe, acc_iter_adv_pcfg_safe, acc_idx1_iter_adv_safe, acc_idx2_iter_adv_safe, acc_comb1_iter_adv_safe, acc_comb2_iter_adv_safe, acc_comb_iter_adv_safe = self.get_log(out_safe, y_safe_cap_adv, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_adv_unsafe, acc_iter_adv_unsafe, acc_iter_adv_pcfg_unsafe, acc_idx1_iter_adv_unsafe, acc_idx2_iter_adv_unsafe, acc_comb1_iter_adv_unsafe, acc_comb2_iter_adv_unsafe, acc_comb_iter_adv_unsafe = self.get_log(out_unsafe, y_unsafe_cap_adv, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_adv_unsafe_target, acc_iter_adv_unsafe_target, acc_iter_adv_pcfg_unsafe_target, acc_idx1_iter_adv_unsafe_target, acc_idx2_iter_adv_unsafe_target, acc_comb1_iter_adv_unsafe_target, acc_comb2_iter_adv_unsafe_target, acc_comb_iter_adv_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target_adv, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)


            ########## universal adversarial attack #####################
            
            x_unsafe_cap_target_adv, y_unsafe_cap_target_adv = self.adv_attack(model, x_safe_cap, x_unsafe_cap, x_unsafe_cap_target, y_safe_cap, y_unsafe_cap, y_unsafe_cap_target,mask, start_idx, end_idx, idx, idx1, idx2, idx3, univ_adv=True, steps=100)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target_adv, y_unsafe_cap_target_adv, mask, attack='emb_in', emb=x_unsafe_cap_target_adv)

            start_idx = start_idx + self.config.threat_count_adv
            end_idx[0] = end_idx[0] + 2*self.config.threat_count_adv
            end_idx[1] = end_idx[1] + 3*self.config.threat_count_adv

            loss_iter_adv_unsafe_target_univ, acc_iter_adv_unsafe_target_univ, acc_iter_adv_pcfg_unsafe_target_univ, acc_idx1_iter_adv_unsafe_target_univ, acc_idx2_iter_adv_unsafe_target_univ, acc_comb1_iter_adv_unsafe_target_univ, acc_comb2_iter_adv_unsafe_target_univ, acc_comb_iter_adv_unsafe_target_univ = self.get_log(out_unsafe_target, y_unsafe_cap_target_adv, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)


            iter_counter+=1

            loss_iter_dict['loss_unsafe_std'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_unsafe_std'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_unsafe_pcfg_std'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_unsafe_idx1_std'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_unsafe_idx2_std'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_unsafe_comb1_std'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_unsafe_comb2_std'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_unsafe_comb_std'][0]+=acc_comb_iter_safe


            loss_iter_dict['loss_unsafe_std'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_unsafe_std'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_unsafe_idx1_std'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_unsafe_idx2_std'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_unsafe_comb1_std'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_unsafe_comb2_std'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_unsafe_comb_std'][1]+=acc_comb_iter_unsafe



            loss_iter_dict['loss_unsafe_std'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_unsafe_std'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_unsafe_idx1_std'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_unsafe_idx2_std'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_unsafe_comb1_std'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_unsafe_comb2_std'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_unsafe_comb_std'][2]+=acc_comb_iter_unsafe_target


            loss_iter_dict['loss_unsafe_adv'][0]+=loss_iter_adv_safe.item()
            acc_iter_dict['acc_unsafe_adv'][0]+=acc_iter_adv_safe
            acc_pcfg_iter_dict['acc_unsafe_pcfg_adv'][0]+=acc_iter_adv_pcfg_safe
            acc_idx1_iter_dict['acc_unsafe_idx1_adv'][0]+=acc_idx1_iter_adv_safe
            acc_idx2_iter_dict['acc_unsafe_idx2_adv'][0]+=acc_idx2_iter_adv_safe
            acc_comb1_iter_dict['acc_unsafe_comb1_adv'][0]+=acc_comb1_iter_adv_safe
            acc_comb2_iter_dict['acc_unsafe_comb2_adv'][0]+=acc_comb2_iter_adv_safe
            acc_comb_iter_dict['acc_unsafe_comb_adv'][0]+=acc_comb_iter_adv_safe


            loss_iter_dict['loss_unsafe_adv'][1]+=loss_iter_adv_unsafe.item()
            acc_iter_dict['acc_unsafe_adv'][1]+=acc_iter_adv_unsafe
            acc_pcfg_iter_dict['acc_unsafe_pcfg_adv'][1]+=acc_iter_adv_pcfg_unsafe
            acc_idx1_iter_dict['acc_unsafe_idx1_adv'][1]+=acc_idx1_iter_adv_unsafe
            acc_idx2_iter_dict['acc_unsafe_idx2_adv'][1]+=acc_idx2_iter_adv_unsafe
            acc_comb1_iter_dict['acc_unsafe_comb1_adv'][1]+=acc_comb1_iter_adv_unsafe
            acc_comb2_iter_dict['acc_unsafe_comb2_adv'][1]+=acc_comb2_iter_adv_unsafe
            acc_comb_iter_dict['acc_unsafe_comb_adv'][1]+=acc_comb_iter_adv_unsafe



            loss_iter_dict['loss_unsafe_adv'][2]+=loss_iter_adv_unsafe_target.item()
            acc_iter_dict['acc_unsafe_adv'][2]+=acc_iter_adv_unsafe_target
            acc_pcfg_iter_dict['acc_unsafe_pcfg_adv'][2]+=acc_iter_adv_pcfg_unsafe_target
            acc_idx1_iter_dict['acc_unsafe_idx1_adv'][2]+=acc_idx1_iter_adv_unsafe_target
            acc_idx2_iter_dict['acc_unsafe_idx2_adv'][2]+=acc_idx2_iter_adv_unsafe_target
            acc_comb1_iter_dict['acc_unsafe_comb1_adv'][2]+=acc_comb1_iter_adv_unsafe_target
            acc_comb2_iter_dict['acc_unsafe_comb2_adv'][2]+=acc_comb2_iter_adv_unsafe_target
            acc_comb_iter_dict['acc_unsafe_comb_adv'][2]+=acc_comb_iter_adv_unsafe_target


            loss_iter_dict['loss_unsafe_univ_adv'][2]+=loss_iter_adv_unsafe_target_univ.item()
            acc_iter_dict['acc_unsafe_univ_adv'][2]+=acc_iter_adv_unsafe_target_univ
            acc_pcfg_iter_dict['acc_unsafe_pcfg_univ_adv'][2]+=acc_iter_adv_pcfg_unsafe_target_univ
            acc_idx1_iter_dict['acc_unsafe_idx1_univ_adv'][2]+=acc_idx1_iter_adv_unsafe_target_univ
            acc_idx2_iter_dict['acc_unsafe_idx2_univ_adv'][2]+=acc_idx2_iter_adv_unsafe_target_univ
            acc_comb1_iter_dict['acc_unsafe_comb1_univ_adv'][2]+=acc_comb1_iter_adv_unsafe_target_univ
            acc_comb2_iter_dict['acc_unsafe_comb2_univ_adv'][2]+=acc_comb2_iter_adv_unsafe_target_univ
            acc_comb_iter_dict['acc_unsafe_comb_univ_adv'][2]+=acc_comb_iter_adv_unsafe_target_univ


            if iter_counter>=max_iterations:
                break


        iter_counter=0

        for batch in test_loader_intermediate:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)

            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_jail_co'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_jail_co'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_pcfg_jail_co'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_idx1_jail_co'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_idx2_jail_co'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_comb1_jail_co'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_comb2_jail_co'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_comb_jail_co'][0]+=acc_comb_iter_safe

            loss_iter_dict['loss_jail_co'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_jail_co'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_idx1_jail_co'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_idx2_jail_co'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_comb1_jail_co'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_comb2_jail_co'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_comb_jail_co'][1]+=acc_comb_iter_unsafe

            loss_iter_dict['loss_jail_co'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_jail_co'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_idx1_jail_co'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_idx2_jail_co'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_comb1_jail_co'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_comb2_jail_co'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_comb_jail_co'][2]+=acc_comb_iter_unsafe_target

            if iter_counter>=max_iterations:
                break


        iter_counter=0

        for batch in test_loader_duplicate:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)


            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_jail_mg_tokens'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_jail_mg_tokens'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_pcfg_jail_mg_tokens'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_idx1_jail_mg_tokens'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_idx2_jail_mg_tokens'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_comb1_jail_mg_tokens'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_comb2_jail_mg_tokens'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_comb_jail_mg_tokens'][0]+=acc_comb_iter_safe

            loss_iter_dict['loss_jail_mg_tokens'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_jail_mg_tokens'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_idx1_jail_mg_tokens'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_idx2_jail_mg_tokens'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_comb1_jail_mg_tokens'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_comb2_jail_mg_tokens'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_comb_jail_mg_tokens'][1]+=acc_comb_iter_unsafe

            loss_iter_dict['loss_jail_mg_tokens'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_jail_mg_tokens'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_idx1_jail_mg_tokens'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_idx2_jail_mg_tokens'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_comb1_jail_mg_tokens'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_comb2_jail_mg_tokens'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_comb_jail_mg_tokens'][2]+=acc_comb_iter_unsafe_target

            if iter_counter>=max_iterations:
                break


        iter_counter=0

        for batch in test_loader_ood_mg:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)

            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_jail_mg_text'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_jail_mg_text'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_pcfg_jail_mg_text'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_idx1_jail_mg_text'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_idx2_jail_mg_text'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_comb1_jail_mg_text'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_comb2_jail_mg_text'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_comb_jail_mg_text'][0]+=acc_comb_iter_safe

            loss_iter_dict['loss_jail_mg_text'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_jail_mg_text'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_idx1_jail_mg_text'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_idx2_jail_mg_text'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_comb1_jail_mg_text'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_comb2_jail_mg_text'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_comb_jail_mg_text'][1]+=acc_comb_iter_unsafe

            loss_iter_dict['loss_jail_mg_text'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_jail_mg_text'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_idx1_jail_mg_text'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_idx2_jail_mg_text'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_comb1_jail_mg_text'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_comb2_jail_mg_text'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_comb_jail_mg_text'][2]+=acc_comb_iter_unsafe_target

            if iter_counter>=max_iterations:
                break




        iter_counter=0

        for batch in test_loader_id_jailbreak_direct:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)

            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_jail_if_tkns_unsafe'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_jail_if_tkns_unsafe'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_pcfg_jail_if_tkns_unsafe'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_idx1_jail_if_tkns_unsafe'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_idx2_jail_if_tkns_unsafe'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_comb1_jail_if_tkns_unsafe'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_comb2_jail_if_tkns_unsafe'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_comb_jail_if_tkns_unsafe'][0]+=acc_comb_iter_safe

            loss_iter_dict['loss_jail_if_tkns_unsafe'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_jail_if_tkns_unsafe'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_idx1_jail_if_tkns_unsafe'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_idx2_jail_if_tkns_unsafe'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_comb1_jail_if_tkns_unsafe'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_comb2_jail_if_tkns_unsafe'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_comb_jail_if_tkns_unsafe'][1]+=acc_comb_iter_unsafe

            loss_iter_dict['loss_jail_if_tkns_unsafe'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_jail_if_tkns_unsafe'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_idx1_jail_if_tkns_unsafe'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_idx2_jail_if_tkns_unsafe'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_comb1_jail_if_tkns_unsafe'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_comb2_jail_if_tkns_unsafe'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_comb_jail_if_tkns_unsafe'][2]+=acc_comb_iter_unsafe_target

            if iter_counter>=max_iterations:
                break



        iter_counter=0

        for batch in test_loader_duplicate2:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)


            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_jail_mg_tokens2'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_jail_mg_tokens2'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_pcfg_jail_mg_tokens2'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_idx1_jail_mg_tokens2'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_idx2_jail_mg_tokens2'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_comb1_jail_mg_tokens2'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_comb2_jail_mg_tokens2'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_comb_jail_mg_tokens2'][0]+=acc_comb_iter_safe

            loss_iter_dict['loss_jail_mg_tokens2'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_jail_mg_tokens2'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_idx1_jail_mg_tokens2'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_idx2_jail_mg_tokens2'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_comb1_jail_mg_tokens2'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_comb2_jail_mg_tokens2'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_comb_jail_mg_tokens2'][1]+=acc_comb_iter_unsafe

            loss_iter_dict['loss_jail_mg_tokens2'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_jail_mg_tokens2'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_idx1_jail_mg_tokens2'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_idx2_jail_mg_tokens2'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_comb1_jail_mg_tokens2'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_comb2_jail_mg_tokens2'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_comb_jail_mg_tokens2'][2]+=acc_comb_iter_unsafe_target

            if iter_counter>=max_iterations:
                break





        iter_counter=0

        for batch in test_loader_safe_jailbreak_direct:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)

            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_jail_if_tkns_safe'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_jail_if_tkns_safe'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_pcfg_jail_if_tkns_safe'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_idx1_jail_if_tkns_safe'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_idx2_jail_if_tkns_safe'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_comb1_jail_if_tkns_safe'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_comb2_jail_if_tkns_safe'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_comb_jail_if_tkns_safe'][0]+=acc_comb_iter_safe

            loss_iter_dict['loss_jail_if_tkns_safe'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_jail_if_tkns_safe'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_idx1_jail_if_tkns_safe'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_idx2_jail_if_tkns_safe'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_comb1_jail_if_tkns_safe'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_comb2_jail_if_tkns_safe'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_comb_jail_if_tkns_safe'][1]+=acc_comb_iter_unsafe

            loss_iter_dict['loss_jail_if_tkns_safe'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_jail_if_tkns_safe'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_idx1_jail_if_tkns_safe'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_idx2_jail_if_tkns_safe'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_comb1_jail_if_tkns_safe'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_comb2_jail_if_tkns_safe'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_comb_jail_if_tkns_safe'][2]+=acc_comb_iter_unsafe_target

            if iter_counter>=max_iterations:
                break


        iter_counter=0

        for batch in test_loader_id_jailbreak_comp:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)

            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_jail_cot_tkns_unsafe'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_jail_cot_tkns_unsafe'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_pcfg_jail_cot_tkns_unsafe'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_unsafe'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_unsafe'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_unsafe'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_unsafe'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_comb_jail_cot_tkns_unsafe'][0]+=acc_comb_iter_safe

            loss_iter_dict['loss_jail_cot_tkns_unsafe'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_jail_cot_tkns_unsafe'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_unsafe'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_unsafe'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_unsafe'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_unsafe'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_comb_jail_cot_tkns_unsafe'][1]+=acc_comb_iter_unsafe

            loss_iter_dict['loss_jail_cot_tkns_unsafe'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_jail_cot_tkns_unsafe'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_unsafe'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_unsafe'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_unsafe'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_unsafe'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_comb_jail_cot_tkns_unsafe'][2]+=acc_comb_iter_unsafe_target

            if iter_counter>=max_iterations:
                break


        iter_counter=0

        for batch in test_loader_safe_jailbreak_comp:
            x_safe_cap, x_safe_cap_target, x_unsafe_cap, x_unsafe_cap_target, _, y_safe_cap, y_safe_cap_target, y_unsafe_cap, y_unsafe_cap_target, _, mask, start_idx, end_idx, idx, idx1, idx2, idx3, idx_clm, safety_label, _ = batch
            x_safe_cap = x_safe_cap.cuda()
            y_safe_cap = y_safe_cap.cuda()
            x_unsafe_cap_target = x_unsafe_cap_target.cuda()
            x_unsafe_cap = x_unsafe_cap.cuda()
            y_unsafe_cap = y_unsafe_cap.cuda()
            y_unsafe_cap_target = y_unsafe_cap_target.cuda()

            mask = mask.cuda()
            end_idx = end_idx.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()

            ################ Standard evaluation #####################
            out_safe, loss_safe = model(x_safe_cap, y_safe_cap, mask)
            out_unsafe, loss_unsafe = model(x_unsafe_cap, y_unsafe_cap, mask)
            out_unsafe_target, loss_unsafe_target = model(x_unsafe_cap_target, y_unsafe_cap_target, mask)

            loss_iter_safe, acc_iter_safe, acc_iter_pcfg_safe, acc_idx1_iter_safe, acc_idx2_iter_safe, acc_comb1_iter_safe, acc_comb2_iter_safe, acc_comb_iter_safe = self.get_log(out_safe, y_safe_cap, loss_safe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe, acc_iter_unsafe, _, acc_idx1_iter_unsafe, acc_idx2_iter_unsafe, acc_comb1_iter_unsafe, acc_comb2_iter_unsafe, acc_comb_iter_unsafe = self.get_log(out_unsafe, y_unsafe_cap, loss_unsafe, start_idx, end_idx, idx, idx1, idx2, idx3)
            loss_iter_unsafe_target, acc_iter_unsafe_target, _, acc_idx1_iter_unsafe_target, acc_idx2_iter_unsafe_target, acc_comb1_iter_unsafe_target, acc_comb2_iter_unsafe_target, acc_comb_iter_unsafe_target = self.get_log(out_unsafe_target, y_unsafe_cap_target, loss_unsafe_target, start_idx, end_idx, idx, idx1, idx2, idx3)

            iter_counter+=1

            loss_iter_dict['loss_jail_cot_tkns_safe'][0]+=loss_iter_safe.item()
            acc_iter_dict['acc_jail_cot_tkns_safe'][0]+=acc_iter_safe
            acc_pcfg_iter_dict['acc_pcfg_jail_cot_tkns_safe'][0]+=acc_iter_pcfg_safe
            acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_safe'][0]+=acc_idx1_iter_safe
            acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_safe'][0]+=acc_idx2_iter_safe
            acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_safe'][0]+=acc_comb1_iter_safe
            acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_safe'][0]+=acc_comb2_iter_safe
            acc_comb_iter_dict['acc_comb_jail_cot_tkns_safe'][0]+=acc_comb_iter_safe

            loss_iter_dict['loss_jail_cot_tkns_safe'][1]+=loss_iter_unsafe.item()
            acc_iter_dict['acc_jail_cot_tkns_safe'][1]+=acc_iter_unsafe
            acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_safe'][1]+=acc_idx1_iter_unsafe
            acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_safe'][1]+=acc_idx2_iter_unsafe
            acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_safe'][1]+=acc_comb1_iter_unsafe
            acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_safe'][1]+=acc_comb2_iter_unsafe
            acc_comb_iter_dict['acc_comb_jail_cot_tkns_safe'][1]+=acc_comb_iter_unsafe

            loss_iter_dict['loss_jail_cot_tkns_safe'][2]+=loss_iter_unsafe_target.item()
            acc_iter_dict['acc_jail_cot_tkns_safe'][2]+=acc_iter_unsafe_target
            acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_safe'][2]+=acc_idx1_iter_unsafe_target
            acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_safe'][2]+=acc_idx2_iter_unsafe_target
            acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_safe'][2]+=acc_comb1_iter_unsafe_target
            acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_safe'][2]+=acc_comb2_iter_unsafe_target
            acc_comb_iter_dict['acc_comb_jail_cot_tkns_safe'][2]+=acc_comb_iter_unsafe_target

            if iter_counter>=max_iterations:
                break



        model.train()

        for i in range(3):
            loss_iter_dict['loss_safe_std'][i]/=iter_counter
            acc_iter_dict['acc_safe_std'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_safe_pcfg_std'][i]/=iter_counter
            acc_idx1_iter_dict['acc_safe_idx1_std'][i]/=iter_counter
            acc_idx2_iter_dict['acc_safe_idx2_std'][i]/=iter_counter
            acc_comb1_iter_dict['acc_safe_comb1_std'][i]/=iter_counter
            acc_comb2_iter_dict['acc_safe_comb2_std'][i]/=iter_counter
            acc_comb_iter_dict['acc_safe_comb_std'][i]/=iter_counter

            loss_iter_dict['loss_safe_adv'][i]/=iter_counter
            acc_iter_dict['acc_safe_adv'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_safe_pcfg_adv'][i]/=iter_counter
            acc_idx1_iter_dict['acc_safe_idx1_adv'][i]/=iter_counter
            acc_idx2_iter_dict['acc_safe_idx2_adv'][i]/=iter_counter
            acc_comb1_iter_dict['acc_safe_comb1_adv'][i]/=iter_counter
            acc_comb2_iter_dict['acc_safe_comb2_adv'][i]/=iter_counter
            acc_comb_iter_dict['acc_safe_comb_adv'][i]/=iter_counter        

            loss_iter_dict['loss_unsafe_std'][i]/=iter_counter
            acc_iter_dict['acc_unsafe_std'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_unsafe_pcfg_std'][i]/=iter_counter
            acc_idx1_iter_dict['acc_unsafe_idx1_std'][i]/=iter_counter
            acc_idx2_iter_dict['acc_unsafe_idx2_std'][i]/=iter_counter
            acc_comb1_iter_dict['acc_unsafe_comb1_std'][i]/=iter_counter
            acc_comb2_iter_dict['acc_unsafe_comb2_std'][i]/=iter_counter
            acc_comb_iter_dict['acc_unsafe_comb_std'][i]/=iter_counter

            loss_iter_dict['loss_unsafe_adv'][i]/=iter_counter
            acc_iter_dict['acc_unsafe_adv'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_unsafe_pcfg_adv'][i]/=iter_counter
            acc_idx1_iter_dict['acc_unsafe_idx1_adv'][i]/=iter_counter
            acc_idx2_iter_dict['acc_unsafe_idx2_adv'][i]/=iter_counter
            acc_comb1_iter_dict['acc_unsafe_comb1_adv'][i]/=iter_counter
            acc_comb2_iter_dict['acc_unsafe_comb2_adv'][i]/=iter_counter
            acc_comb_iter_dict['acc_unsafe_comb_adv'][i]/=iter_counter


            loss_iter_dict['loss_jail_mg_text'][i]/=iter_counter
            acc_iter_dict['acc_jail_mg_text'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_pcfg_jail_mg_text'][i]/=iter_counter
            acc_idx1_iter_dict['acc_idx1_jail_mg_text'][i]/=iter_counter
            acc_idx2_iter_dict['acc_idx2_jail_mg_text'][i]/=iter_counter
            acc_comb1_iter_dict['acc_comb1_jail_mg_text'][i]/=iter_counter
            acc_comb2_iter_dict['acc_comb2_jail_mg_text'][i]/=iter_counter
            acc_comb_iter_dict['acc_comb_jail_mg_text'][i]/=iter_counter
        
            loss_iter_dict['loss_jail_mg_tokens'][i]/=iter_counter
            acc_iter_dict['acc_jail_mg_tokens'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_pcfg_jail_mg_tokens'][i]/=iter_counter
            acc_idx1_iter_dict['acc_idx1_jail_mg_tokens'][i]/=iter_counter
            acc_idx2_iter_dict['acc_idx2_jail_mg_tokens'][i]/=iter_counter
            acc_comb1_iter_dict['acc_comb1_jail_mg_tokens'][i]/=iter_counter
            acc_comb2_iter_dict['acc_comb2_jail_mg_tokens'][i]/=iter_counter
            acc_comb_iter_dict['acc_comb_jail_mg_tokens'][i]/=iter_counter

            loss_iter_dict['loss_jail_co'][i]/=iter_counter
            acc_iter_dict['acc_jail_co'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_pcfg_jail_co'][i]/=iter_counter
            acc_idx1_iter_dict['acc_idx1_jail_co'][i]/=iter_counter
            acc_idx2_iter_dict['acc_idx2_jail_co'][i]/=iter_counter
            acc_comb1_iter_dict['acc_comb1_jail_co'][i]/=iter_counter
            acc_comb2_iter_dict['acc_comb2_jail_co'][i]/=iter_counter
            acc_comb_iter_dict['acc_comb_jail_co'][i]/=iter_counter


            loss_iter_dict['loss_safe_univ_adv'][i]/=iter_counter
            acc_iter_dict['acc_safe_univ_adv'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_safe_pcfg_univ_adv'][i]/=iter_counter
            acc_idx1_iter_dict['acc_safe_idx1_univ_adv'][i]/=iter_counter
            acc_idx2_iter_dict['acc_safe_idx2_univ_adv'][i]/=iter_counter
            acc_comb1_iter_dict['acc_safe_comb1_univ_adv'][i]/=iter_counter
            acc_comb2_iter_dict['acc_safe_comb2_univ_adv'][i]/=iter_counter
            acc_comb_iter_dict['acc_safe_comb_univ_adv'][i]/=iter_counter        

            loss_iter_dict['loss_unsafe_univ_adv'][i]/=iter_counter
            acc_iter_dict['acc_unsafe_univ_adv'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_unsafe_pcfg_univ_adv'][i]/=iter_counter
            acc_idx1_iter_dict['acc_unsafe_idx1_univ_adv'][i]/=iter_counter
            acc_idx2_iter_dict['acc_unsafe_idx2_univ_adv'][i]/=iter_counter
            acc_comb1_iter_dict['acc_unsafe_comb1_univ_adv'][i]/=iter_counter
            acc_comb2_iter_dict['acc_unsafe_comb2_univ_adv'][i]/=iter_counter
            acc_comb_iter_dict['acc_unsafe_comb_univ_adv'][i]/=iter_counter
        
            loss_iter_dict['loss_jail_if_tkns_safe'][i]/=iter_counter
            acc_iter_dict['acc_jail_if_tkns_safe'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_pcfg_jail_if_tkns_safe'][i]/=iter_counter
            acc_idx1_iter_dict['acc_idx1_jail_if_tkns_safe'][i]/=iter_counter
            acc_idx2_iter_dict['acc_idx2_jail_if_tkns_safe'][i]/=iter_counter
            acc_comb1_iter_dict['acc_comb1_jail_if_tkns_safe'][i]/=iter_counter
            acc_comb2_iter_dict['acc_comb2_jail_if_tkns_safe'][i]/=iter_counter
            acc_comb_iter_dict['acc_comb_jail_if_tkns_safe'][i]/=iter_counter
        

        
            loss_iter_dict['loss_jail_mg_tokens2'][i]/=iter_counter
            acc_iter_dict['acc_jail_mg_tokens2'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_pcfg_jail_mg_tokens2'][i]/=iter_counter
            acc_idx1_iter_dict['acc_idx1_jail_mg_tokens2'][i]/=iter_counter
            acc_idx2_iter_dict['acc_idx2_jail_mg_tokens2'][i]/=iter_counter
            acc_comb1_iter_dict['acc_comb1_jail_mg_tokens2'][i]/=iter_counter
            acc_comb2_iter_dict['acc_comb2_jail_mg_tokens2'][i]/=iter_counter
            acc_comb_iter_dict['acc_comb_jail_mg_tokens2'][i]/=iter_counter


            loss_iter_dict['loss_jail_if_tkns_unsafe'][i]/=iter_counter
            acc_iter_dict['acc_jail_if_tkns_unsafe'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_pcfg_jail_if_tkns_unsafe'][i]/=iter_counter
            acc_idx1_iter_dict['acc_idx1_jail_if_tkns_unsafe'][i]/=iter_counter
            acc_idx2_iter_dict['acc_idx2_jail_if_tkns_unsafe'][i]/=iter_counter
            acc_comb1_iter_dict['acc_comb1_jail_if_tkns_unsafe'][i]/=iter_counter
            acc_comb2_iter_dict['acc_comb2_jail_if_tkns_unsafe'][i]/=iter_counter
            acc_comb_iter_dict['acc_comb_jail_if_tkns_unsafe'][i]/=iter_counter

            loss_iter_dict['loss_jail_cot_tkns_safe'][i]/=iter_counter
            acc_iter_dict['acc_jail_cot_tkns_safe'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_pcfg_jail_cot_tkns_safe'][i]/=iter_counter
            acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_safe'][i]/=iter_counter
            acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_safe'][i]/=iter_counter
            acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_safe'][i]/=iter_counter
            acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_safe'][i]/=iter_counter
            acc_comb_iter_dict['acc_comb_jail_cot_tkns_safe'][i]/=iter_counter
        
            loss_iter_dict['loss_jail_cot_tkns_unsafe'][i]/=iter_counter
            acc_iter_dict['acc_jail_cot_tkns_unsafe'][i]/=iter_counter
            acc_pcfg_iter_dict['acc_pcfg_jail_cot_tkns_unsafe'][i]/=iter_counter
            acc_idx1_iter_dict['acc_idx1_jail_cot_tkns_unsafe'][i]/=iter_counter
            acc_idx2_iter_dict['acc_idx2_jail_cot_tkns_unsafe'][i]/=iter_counter
            acc_comb1_iter_dict['acc_comb1_jail_cot_tkns_unsafe'][i]/=iter_counter
            acc_comb2_iter_dict['acc_comb2_jail_cot_tkns_unsafe'][i]/=iter_counter
            acc_comb_iter_dict['acc_comb_jail_cot_tkns_unsafe'][i]/=iter_counter

        return loss_iter_dict, acc_iter_dict, acc_pcfg_iter_dict, acc_idx1_iter_dict, acc_idx2_iter_dict,  acc_comb1_iter_dict, acc_comb2_iter_dict, acc_comb_iter_dict