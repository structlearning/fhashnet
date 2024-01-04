import argparse
import random
import time
import pickle
import sys
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = False
import torch.nn as nn
import numpy as np
from common import logger, set_log
from src.utils import cudavar
from src.hashing_main import *
#from src.graph_data_trainer import pairwise_ranking_loss_similarity
#from src.locality_sensitive_hashing import *
#import matplotlib.pyplot as plt
#from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import tqdm
from datetime import datetime
from src.earlystopping import EarlyStoppingModule
import math
from sklearn.utils import shuffle

def pairwise_ranking_loss_similarity(predPos, predNeg, margin):

    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = margin + expanded_2 - expanded_1
    hinge = torch.nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(n_1*n_2)

def sigmoid_pair_hinge_sim(av,a, b):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the asym hinge similarity -max(0,a[i]- b[j]) for all i and j.
    :return: Matrix with res[i][j]  = -max(0,a[i]- b[j])
    """
#     sig_in = torch.tensor((np.maximum((a[:,None,:]-b[None,:,:]),0)).sum(-1),dtype=torch.float64)
    sig_in = torch.tensor((np.maximum((a[:,None,:]-b[None,:,:]),0)).sum(-1))
    sig_out =  torch.nn.Sigmoid()(av.sigmoid_a*sig_in + av.sigmoid_b)
    return sig_out.detach().numpy()

def sigmoid_hinge_sim(av,a, b):
    """
    input dim: a (m x d), b (m x d)
    output dim: m 
    Computes the asym hinge similarity -max(0,a[i]- b[i]) for all i.
    """
    #sig_in = torch.maximum(a-b,torch.tensor([0])).sum(-1) 
    sig_in = (torch.nn.ReLU()(a-b) ).sum(-1) 
    sig_out =  torch.nn.Sigmoid()(av.sigmoid_a*sig_in + av.sigmoid_b)
    return sig_out.detach().squeeze()


class HashCodeTrainer(nn.Module):
    """
      Fetch embeddings hq, hc 
      Generate fmaps - q+c 
      feed into NN(Linear+tanh)
      Compute loss on hashcode  
    """
    def __init__(self, av):
      super(HashCodeTrainer, self).__init__()
      self.av = av
      

      if self.av.HASH_MODE=="flora": 
        self.init_qnet = []
        self.inner_hs = [self.av.embed_dim] +  [self.av.flora_dim]
        for h0, h1 in zip(self.inner_hs, self.inner_hs[1:]):
              lin = torch.nn.Linear(h0, h1)
              if self.av.INIT_GAUSS:
                  nn.init.normal_(lin.weight)
              self.init_qnet.append(lin)
              if av.LAYER_NORMALIZE:
                self.init_qnet.append(torch.nn.LayerNorm(h1))
              self.init_qnet.append(torch.nn.ReLU())
        self.init_qnet.pop() # pop the last relu/tanh

        self.init_qnet = torch.nn.Sequential(*self.init_qnet)
        
        self.init_cnet = []
        for h0, h1 in zip(self.inner_hs, self.inner_hs[1:]):
           lin = torch.nn.Linear(h0, h1)
           if self.av.INIT_GAUSS:
               nn.init.normal_(lin.weight)
           self.init_cnet.append(lin)
           if av.LAYER_NORMALIZE:
             self.init_cnet.append(torch.nn.LayerNorm(h1))
           self.init_cnet.append(torch.nn.ReLU())
        self.init_cnet.pop() # pop the last relu/tanh
        
        self.init_cnet = torch.nn.Sequential(*self.init_cnet)



      self.init_net = []
      if self.av.FMAP_SELECT == -1:
        #self.inner_hs = [self.av.m_use * self.av.embed_dim * 4] + av.HIDDEN_LAYERS + [self.av.hcode_dim]
        if self.av.HASH_MODE=="dot":
            self.inner_hs = [self.av.embed_dim+1] + av.HIDDEN_LAYERS + [self.av.hcode_dim]
        elif self.av.HASH_MODE=="dot2":
            self.inner_hs = [self.av.embed_dim+2] + av.HIDDEN_LAYERS + [self.av.hcode_dim]
        elif self.av.HASH_MODE=="flora":
            self.inner_hs = [self.av.flora_dim] + av.HIDDEN_LAYERS + [self.av.hcode_dim]
        else:
            self.inner_hs = [self.av.embed_dim] + av.HIDDEN_LAYERS + [self.av.hcode_dim]
      else:
        self.inner_hs = [self.av.FMAP_SELECT] + av.HIDDEN_LAYERS + [self.av.hcode_dim]
      for h0, h1 in zip(self.inner_hs, self.inner_hs[1:]):
            lin = torch.nn.Linear(h0, h1)
            if self.av.INIT_GAUSS:
                nn.init.normal_(lin.weight)
            self.init_net.append(lin)
            if av.LAYER_NORMALIZE:
              self.init_net.append(torch.nn.LayerNorm(h1))
            self.init_net.append(torch.nn.ReLU())
      #Note: there will always be a last relu
      self.init_net.pop() # pop the last relu
      self.init_net = torch.nn.Sequential(*self.init_net)
      self.tanh  = nn.Tanh()
      
      #nn.init.normal_(self.hash_linear1.weight)

    def forward(self, fmaps,isQ=True):
        """
            :param  Fmaps
            :return  Hcodes
        """
        if self.av.LOSS_TYPE == "permgnn_loss_hingeemb" or self.av.LOSS_TYPE == "sc_loss_hingeemb" :
            code = self.init_net(cudavar(self.av,fmaps))
            return self.tanh(self.av.TANH_TEMP * code)
        elif self.av.LOSS_TYPE == "flora_hingeemb" or self.av.LOSS_TYPE == "flora_hingeemb2" :
            if isQ:
                code = self.init_net(self.init_qnet(cudavar(self.av,fmaps)))
            else:    
                code = self.init_net(self.init_cnet(cudavar(self.av,fmaps)))
            return self.tanh(self.av.TANH_TEMP * code)
        else:
            code = self.init_net(cudavar(self.av,fmaps))
            return  code/torch.norm(code,dim=-1,keepdim=True)


    def computeLoss(self, cfmaps, qfmaps, targets):
      """
        :param   cfmaps  : corpus fourier maps
        :param   qfmaps  : query fourier maps
        :param   targets : ground truth scores 0/1
        :return  loss   : Hinge ranking loss
      """
      #q_hcodes = self.forward(qfmaps.cuda())
      #c_hcodes = self.forward(cfmaps.cuda())
      if self.av.LOSS_TYPE == "permgnn_loss_hingeemb":
          #Note : here targets and qfmaps are None  
        all_hcodes = self.forward(cfmaps.cuda())
        bit_balance_loss = torch.sum(torch.abs(torch.sum(all_hcodes,dim=0)))/(all_hcodes.shape[0]*all_hcodes.shape[1])
        decorrelation_loss = torch.abs(torch.mean((all_hcodes.T@all_hcodes).fill_diagonal_(0)))
        fence_sitting_loss =  torch.norm(all_hcodes.abs()-1, p=1)/ (all_hcodes.shape[0]*all_hcodes.shape[1])
        loss = self.av.FENCE_LAMBDA * fence_sitting_loss +\
               self.av.DECORR_LAMBDA * decorrelation_loss+\
               (1-self.av.FENCE_LAMBDA-self.av.DECORR_LAMBDA) * bit_balance_loss
        return loss, bit_balance_loss,decorrelation_loss, fence_sitting_loss
      elif self.av.LOSS_TYPE == "sc_loss_hingeemb":
          #Note : here targets and qfmaps are Non#e  
        q_hcodes = self.forward(qfmaps.cuda())
        c_hcodes = self.forward(cfmaps.cuda())
        all_hcodes = torch.cat([q_hcodes,c_hcodes])
        
        fence_sitting_loss =  torch.norm(all_hcodes.abs()-1, p=1)/ (all_hcodes.shape[0]*all_hcodes.shape[1])
        bit_balance_loss = torch.sum(torch.abs(torch.sum(all_hcodes,dim=0)))/(all_hcodes.shape[0]*all_hcodes.shape[1])

        preds = (q_hcodes*c_hcodes).sum(-1)
        predPos = preds[targets>0.5]
        predNeg = preds[targets<0.5]

        ranking_loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.av.SCLOSS_MARGIN)
        loss = self.av.FENCE_LAMBDA * fence_sitting_loss +\
               self.av.C1_LAMBDA * ranking_loss+\
               (1-self.av.FENCE_LAMBDA-self.av.C1_LAMBDA) * bit_balance_loss
        return loss, bit_balance_loss,ranking_loss, fence_sitting_loss
      elif self.av.LOSS_TYPE == "flora_hingeemb":
          #Note : here targets and qfmaps are Non#e  
        q_hcodes = self.forward(qfmaps.cuda(), isQ=True)
        c_hcodes = self.forward(cfmaps.cuda(),isQ=False)
        all_hcodes = torch.cat([q_hcodes,c_hcodes])

        preds = ( (q_hcodes*c_hcodes).sum(-1)/(2*self.av.hcode_dim) ) + 0.5
        scores = sigmoid_hinge_sim(self.av,qfmaps,cfmaps)
        score_loss= torch.mean(torch.abs(preds-scores))

        bit_balance_loss = torch.sum(torch.abs(torch.sum(all_hcodes,dim=0)))/(all_hcodes.shape[0]*all_hcodes.shape[1])

        prod = self.init_net[0].weight@self.init_net[0].weight.T
        eye = cudavar(self.av,torch.eye(len(prod)))
        decorr_loss = torch.norm(prod-eye)

        loss = self.av.FENCE_LAMBDA * bit_balance_loss +\
               self.av.C1_LAMBDA * decorr_loss+\
               (1-self.av.FENCE_LAMBDA-self.av.C1_LAMBDA) * score_loss
        return loss, bit_balance_loss,decorr_loss, score_loss
      elif self.av.LOSS_TYPE == "flora_hingeemb2":
          #Note : here targets and qfmaps are Non#e  
        q_hcodes = self.forward(qfmaps.cuda(), isQ=True)
        c_hcodes = self.forward(cfmaps.cuda(),isQ=False)
        all_hcodes = torch.cat([q_hcodes,c_hcodes])

        preds = ( (q_hcodes*c_hcodes).sum(-1)/(2*self.av.hcode_dim) ) + 0.5
        #scores = sigmoid_hinge_sim(self.av,qfmaps,cfmaps)
        score_loss= torch.mean(torch.abs(preds-targets))

        bit_balance_loss = torch.sum(torch.abs(torch.sum(all_hcodes,dim=0)))/(all_hcodes.shape[0]*all_hcodes.shape[1])

        prod = self.init_net[0].weight@self.init_net[0].weight.T
        eye = cudavar(self.av,torch.eye(len(prod)))
        decorr_loss = torch.norm(prod-eye)

        loss = self.av.FENCE_LAMBDA * bit_balance_loss +\
               self.av.C1_LAMBDA * decorr_loss+\
               (1-self.av.FENCE_LAMBDA-self.av.C1_LAMBDA) * score_loss
        return loss, bit_balance_loss,decorr_loss, score_loss
      elif self.av.LOSS_TYPE == "cos_ap":
        q_hcodes = self.forward(qfmaps.cuda())
        c_hcodes = self.forward(cfmaps.cuda())
        assert False , print(f"This has now been deprecated. Use dot_ap w/o tanh")
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        preds = cos(q_hcodes, c_hcodes)
        predPos = preds[targets>0.5]
        predNeg = preds[targets<0.5]
        loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.av.MARGIN)
      elif self.av.LOSS_TYPE == "dot_ap":
        q_hcodes = self.forward(qfmaps.cuda())
        c_hcodes = self.forward(cfmaps.cuda())
        if self.av.NO_TANH:
            q_pred = q_hcodes
            c_pred = c_hcodes
            fence_sitting_loss = 0 
        else:    
            q_pred = self.tanh(self.av.TANH_TEMP * q_hcodes)
            c_pred = self.tanh(self.av.TANH_TEMP * c_hcodes)
            fence_sitting_loss = (torch.norm(q_pred.abs()-1, p=1) + torch.norm(c_pred.abs()-1, p=1))/ (2*q_pred.shape[0])
        preds = (q_pred*c_pred).sum(-1)  
        predPos = preds[targets>0.5]
        predNeg = preds[targets<0.5]
        loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.av.MARGIN)
        loss = (1-self.av.FENCE_LAMBDA)*loss + self.av.FENCE_LAMBDA * fence_sitting_loss
      elif self.av.LOSS_TYPE == "dot_ce":
        assert False , print(f"This has now been deprecated. Discuss and implement new verison if reqd.")
        dot_prod = q_hcodes*c_hcodes
        pos_dot = dot_prod[targets>0.5]
        neg_dot = dot_prod[targets<0.5]
        loss = ((pos_dot-1).abs().sum() + (neg_dot+1).abs().sum())/cfmaps.shape[0]
      else:
        return NotImplementedError() 

      return loss


class FmapData(object):
    def __init__(self, av): 
        self.av = av
        self.lsh = LSH(self.av)
        corpus_embeds_fetch_start = time.time()
        self.corpus_embeds = torch.from_numpy(fetch_corpus_embeddings(self.av))
        corpus_embeds_fetch_time = time.time() - corpus_embeds_fetch_start
        logger.info(f"Corpus embeds shape: {self.corpus_embeds.shape}, time={corpus_embeds_fetch_time}")
        
        #if self.av.HASH_MODE=="dot":
        self.corpus_embeds_aug = torch.zeros((self.corpus_embeds.shape[0], self.corpus_embeds.shape[1]+1),dtype = self.corpus_embeds.dtype)
        max_norm = np.max(np.linalg.norm(self.corpus_embeds, axis=1))
        batch_sz = 40000
        for i in range(0, self.corpus_embeds.shape[0],batch_sz):
            batch_item =  self.corpus_embeds[i:i+batch_sz]
            batch_item_scaled = batch_item/max_norm
            app = np.expand_dims(np.sqrt(1-np.square(np.linalg.norm(batch_item_scaled, axis=1))),axis=-1)
            self.corpus_embeds_aug[i:i+batch_sz] = torch.from_numpy(np.hstack((batch_item_scaled,app)))

        
        M = np.linalg.norm(self.corpus_embeds,axis=-1).max()
        self.corpus_embeds_aug2 = torch.zeros((self.corpus_embeds.shape[0], self.corpus_embeds.shape[1]+2),dtype = self.corpus_embeds.dtype)
        batch_sz = 40000
        for i in range(0, self.corpus_embeds.shape[0],batch_sz):
            batch_item =  self.corpus_embeds[i:i+batch_sz]
            zero_np = np.zeros((batch_item.shape[0],1))
            app = np.expand_dims(np.sqrt(np.square(M)-np.square(np.linalg.norm(batch_item, axis=1))),axis=-1)
            self.corpus_embeds_aug2[i:i+batch_sz] = torch.from_numpy(np.hstack((zero_np,batch_item,app))/M)


        corpusfmaps_start_time = time.time()

        #self.batch_sz = 40000
        #fmaps_init = time.time()
        #self.corpus_fmaps = cudavar(self.av,torch.zeros((self.corpus_embeds.shape[0], self.av.m_use * self.corpus_embeds.shape[1] * 4)))
        #logger.info(f"init corpus fmaps, shape={self.corpus_fmaps.shape}, time={time.time()-fmaps_init}")
        #for i in tqdm.tqdm(range(0, self.corpus_embeds.shape[0],self.batch_sz)):
        #    if self.av.SPECIAL_FHASH == "SymCorpus":
        #        self.corpus_fmaps[i:i+self.batch_sz,:] = torch.from_numpy(self.lsh.generate_fmap(self.av, self.corpus_embeds[i:i+self.batch_sz], isQuery=False)).type(torch.float)
        #    elif self.av.SPECIAL_FHASH == "SymQuery":
        #        self.corpus_fmaps[i:i+self.batch_sz,:] = torch.from_numpy(self.lsh.generate_fmap(self.av, self.corpus_embeds[i:i+self.batch_sz], isQuery=True)).type(torch.float)
        #    else:
        #        self.corpus_fmaps[i:i+self.batch_sz,:] = torch.from_numpy(self.lsh.generate_fmap(self.av, self.corpus_embeds[i:i+self.batch_sz], isQuery=False)).type(torch.float)
        #    
        #if not av.FMAP_SELECT == -1:
        #    self.corpus_fmaps = self.corpus_fmaps[:,:self.av.FMAP_SELECT]
        #
        #corpusfmaps_time = time.time() - corpusfmaps_start_time
        #logger.info(f"Corpus fmaps shape: {self.corpus_fmaps.shape}, time={corpusfmaps_time}")
        
        self.query_embeds  = {}
        self.query_embeds_aug  = {}
        self.query_embeds_aug2  = {}
        #self.query_fmaps  = {}
        self.ground_truth = {}
        self.list_pos = {}
        self.list_neg = {} 
        self.list_total_arranged_per_query = {}
        self.labels_total_arranged_per_query = {}
        self.eval_batches = {}
        for mode in ["train", "val"]:
        
            self.av.SPLIT = mode
            self.query_embeds[mode] = fetch_query_embeddings(self.av)
       
            #Query augmentation only needs a zero appended to the end of normalized query vec
            self.query_embeds_aug[mode] = torch.from_numpy(np.zeros((self.query_embeds[mode].shape[0], self.query_embeds[mode].shape[1]+1),dtype = self.query_embeds[mode].dtype))
            self.query_embeds_aug[mode][:,:self.query_embeds[mode].shape[1]] = torch.from_numpy(self.query_embeds[mode]/(np.linalg.norm(self.query_embeds[mode],axis=1)[:,None]+1e-8))


            self.query_embeds_aug2[mode] = torch.from_numpy(np.zeros((self.query_embeds[mode].shape[0], self.query_embeds[mode].shape[1]+2),dtype = self.query_embeds[mode].dtype))
            zero_np = np.zeros((self.query_embeds[mode].shape[0],1))
            app = np.expand_dims(np.sqrt(np.square(M)-np.square(np.linalg.norm(self.query_embeds[mode], axis=1))),axis=-1)
            self.query_embeds_aug2[mode] = torch.from_numpy((np.hstack((app,self.query_embeds[mode],zero_np))/M).astype(dtype = self.query_embeds[mode].dtype))
            
            self.query_embeds[mode] = torch.from_numpy(self.query_embeds[mode])


            #if self.av.SPECIAL_FHASH == "SymCorpus":
            #    self.query_fmaps[mode] = cudavar(self.av,torch.from_numpy(self.lsh.generate_fmap(self.av, self.query_embeds[mode], isQuery=False)).type(torch.float))
            #elif self.av.SPECIAL_FHASH == "SymQuery":
            #    self.query_fmaps[mode] = cudavar(self.av,torch.from_numpy(self.lsh.generate_fmap(self.av, self.query_embeds[mode], isQuery=True)).type(torch.float))
            #else:
            #    self.query_fmaps[mode] = cudavar(self.av,torch.from_numpy(self.lsh.generate_fmap(self.av, self.query_embeds[mode], isQuery=True)).type(torch.float))

            #if not av.FMAP_SELECT == -1:
            #    self.query_fmaps[mode] = self.query_fmaps[mode][:,:self.av.FMAP_SELECT]

            if self.av.LOSS_TYPE == "sc_loss_hingeemb":
                num_pos = int(self.corpus_embeds.shape[0]/(2**self.av.subset_size))
                if self.av.SilverSig:
                    sc = sigmoid_pair_hinge_sim(self.av,self.query_embeds[mode],self.corpus_embeds) 
                else:
                    if self.av.TASK == "cos":
                        sc = cosine_similarity(self.query_embeds[mode],self.corpus_embeds) 
                    elif self.av.TASK == "dot" or self.av.TASK == "dot2":
                        sc = dot_sim(self.query_embeds[mode],self.corpus_embeds) 
                    else:
                        sc = hinge_sim(self.query_embeds[mode],self.corpus_embeds) 
                gt = {}
                for qidx in range(len(self.query_embeds[mode])):
                    pos_cids = torch.flip(torch.argsort(torch.tensor(sc[qidx])), dims=[0])[:num_pos].tolist()
                    gt[qidx] = pos_cids
                
                self.ground_truth[mode] = gt
            else:   
                self.ground_truth[mode] = fetch_ground_truths(self.av)
            self.list_pos[mode] = []
            self.list_neg[mode] = []
            self.list_total_arranged_per_query[mode] = []
            self.labels_total_arranged_per_query[mode] = []
            for q in range(self.query_embeds[mode].shape[0]) :
                for c in range(self.corpus_embeds.shape[0]): 
                    if c in self.ground_truth[mode][q]:
                        self.list_pos[mode].append(((q,c),1.0))
                        self.list_total_arranged_per_query[mode].append(((q,c),1.0))
                        self.labels_total_arranged_per_query[mode].append(1.0)
                    else:
                        self.list_neg[mode].append(((q,c),0.0))  
                        self.list_total_arranged_per_query[mode].append(((q,c),0.0))
                        self.labels_total_arranged_per_query[mode].append(0.0)
            self.eval_batches[mode] = {} 
            
        logger.info('Query embeds fetched and fmaps generated.')
        logger.info('Ground truth fetched.')
        self.preprocess_create_batches()
        self.preprocess_create_per_query_batches()
        self.corpus_embeds_aug2 = self.corpus_embeds_aug2.cuda()
        self.corpus_embeds_aug = self.corpus_embeds_aug.cuda()
        self.corpus_embeds = self.corpus_embeds.cuda()
        for mode in ["train", "val"]:
            self.query_embeds_aug[mode] = self.query_embeds_aug[mode].cuda()
            self.query_embeds_aug2[mode] = self.query_embeds_aug2[mode].cuda()
            self.query_embeds[mode] = self.query_embeds[mode].cuda()

    def create_fmap_batches(self,mode):
        if self.av.HASH_MODE=="dot":
            # all_fmaps = torch.cat([torch.from_numpy(self.query_embeds_aug[mode]), torch.from_numpy(self.corpus_embeds_aug)])
            all_fmaps = torch.cat([self.query_embeds_aug[mode], self.corpus_embeds_aug])
        elif self.av.HASH_MODE=="dot2":
            # all_fmaps = torch.cat([torch.from_numpy(self.query_embeds_aug2[mode]), torch.from_numpy(self.corpus_embeds_aug2)])
            all_fmaps = torch.cat([self.query_embeds_aug2[mode], self.corpus_embeds_aug2])
        else:
            # all_fmaps = torch.cat([torch.from_numpy(self.query_embeds[mode]), torch.from_numpy(self.corpus_embeds)])
            all_fmaps = torch.cat([self.query_embeds[mode], self.corpus_embeds])
        if mode == "train":
            all_fmaps = all_fmaps[torch.randperm(all_fmaps.shape[0])]
        
        self.batches = list(all_fmaps.split(self.av.BATCH_SIZE))
        self.num_batches = len(self.batches)
        return self.num_batches

    def fetch_fmap_batched_data_by_id(self,i):
        assert(i < self.num_batches)  
        return self.batches[i]

    def create_batches(self,list_all,VAL_BATCH_SIZE=10000):
        """
          create batches as is and return number of batches created
        """

        self.batches = []
        self.alists = []
        self.blists = []
        self.scores = []

        pair_all, score_all = zip(* list_all)
        as_all, bs_all = zip(* pair_all)
        for i in range(0, len(list_all), VAL_BATCH_SIZE):
          self.batches.append(list_all[i:i+VAL_BATCH_SIZE])
          self.alists.append(list(as_all[i:i+VAL_BATCH_SIZE]))
          self.blists.append(list(bs_all[i:i+VAL_BATCH_SIZE]))
          self.scores.append(torch.tensor(score_all[i:i+VAL_BATCH_SIZE]).cuda())

     
        self.num_batches = len(self.batches)  

        return self.num_batches


    def create_batches_with_p2n(self,mode):
      """
        Creates shuffled batches while maintaining given ratio
      """
      lpos = self.list_pos[mode]
      lneg = self.list_neg[mode]
      
      random.shuffle(lpos)
      random.shuffle(lneg)

      lpos_pair, lposs = zip(*lpos)
      lposa, lposb = zip(*lpos_pair)

      lneg_pair, lnegs = zip(*lneg)
      lnega, lnegb = zip(*lneg_pair)

      p2n_ratio = self.av.P2N
      batches_pos, batches_neg = [],[]
      as_pos, as_neg, bs_pos, bs_neg, ss_pos, ss_neg = [], [], [], [], [], []
      
      logger.info(f"av.BATCH_SIZE = {self.av.BATCH_SIZE}")
      if self.av.BATCH_SIZE > 0:
        npos = math.ceil((p2n_ratio/(1+p2n_ratio))*self.av.BATCH_SIZE)
        nneg = self.av.BATCH_SIZE-npos
        self.num_batches = int(math.ceil(max(len(lneg) / nneg, len(lpos) / npos)))
        pos_rep = int(math.ceil((npos * self.num_batches / len(lpos))))
        neg_rep = int(math.ceil((nneg * self.num_batches / len(lneg))))
        logger.info(f"Replicating lpos {pos_rep} times, lneg {neg_rep} times")
        lpos = lpos * pos_rep
        lposa = lposa * pos_rep
        lposb = lposb * pos_rep
        lposs = lposs * pos_rep

        lneg = lneg * neg_rep
        lnega = lnega * neg_rep
        lnegb = lnegb * neg_rep
        lnegs = lnegs * neg_rep

        logger.info(f"self.num_batches = {self.num_batches}")

        for i in tqdm.tqdm(range(self.num_batches)):
          try:
            batches_pos.append(lpos[i * npos:(i+1) * npos])
            as_pos.append(lposa[i * npos:(i+1) * npos])
            bs_pos.append(lposb[i * npos:(i+1) * npos])
            ss_pos.append(lposs[i * npos:(i+1) * npos])

            assert len(batches_pos[-1]) > 0
          except Exception as e:
            logger.exception(e, exc_info=True)
            logger.info(batches_pos[-1], len(lpos), (i+1)*npos)

        for i in tqdm.tqdm(range(self.num_batches)):
          try:
            batches_neg.append(lneg[i * nneg:(i+1) * nneg])
            as_neg.append(lnega[i * nneg:(i+1) * nneg])
            bs_neg.append(lnegb[i * nneg:(i+1) * nneg])
            ss_neg.append(lnegs[i * nneg:(i+1) * nneg])
            assert len(batches_neg[-1]) > 0
          except Exception as e:
            logger.exception(e, exc_info=True)
            logger.info(batches_neg[-1], len(lneg), (i+1)*nneg)
      else:
        self.num_batches = 1
        batches_pos.append(lpos)
        batches_neg.append(lneg)
       
      self.batches = [a+b for (a,b) in zip(batches_pos[:self.num_batches],batches_neg[:self.num_batches])]
      self.alists = [list(a+b) for (a,b) in zip(as_pos[:self.num_batches],as_neg[:self.num_batches])]
      self.blists = [list(a+b) for (a,b) in zip(bs_pos[:self.num_batches],bs_neg[:self.num_batches])]
      self.scores = [torch.tensor(a+b).cuda() for (a,b) in zip(ss_pos[:self.num_batches],ss_neg[:self.num_batches])]
      self.alists_tensorized = [torch.tensor(list(a+b)) for (a,b) in zip(as_pos[:self.num_batches],as_neg[:self.num_batches])]
      self.mode = mode

      return self.num_batches

    def preprocess_create_per_query_batches(self):
        split_len  = self.corpus_embeds.shape[0]
        print("In preprocess_create_per_query_batches")
        self.per_query_batches={} 
        for mode in ["train", "val"]:
            self.per_query_batches[mode]={}
            whole_list = self.list_total_arranged_per_query[mode]
            batches = [whole_list[i:i + split_len] for i in range(0, len(whole_list), split_len)]
            alists = []
            blists = []
            scores = []

            for btch in batches:
                btch_np = np.array(btch)
                scores.append(torch.tensor(btch_np[:,1].tolist()).cuda())
                temp = np.array(btch_np[:,0].tolist())
                alists.append(temp[:,0].tolist())
                blists.append(temp[:,1].tolist())
                
            self.per_query_batches[mode]['alists'] = alists
            self.per_query_batches[mode]['blists'] = blists
            self.per_query_batches[mode]['scores'] = scores
            

    def preprocess_create_batches(self,VAL_BATCH_SIZE=10000):
        for mode in ["train", "val"]:
            list_all_ap = self.list_pos[mode] + self.list_neg[mode]
            list_all_map = self.list_total_arranged_per_query[mode]
            label_map = self.labels_total_arranged_per_query[mode]
            self.eval_batches[mode] = {}
            for metric in ["ap", "map"]:
                self.eval_batches[mode][metric] = {}
                list_all = list_all_ap if metric=="ap" else list_all_map
                batches = []
                alists = []
                blists = []
                scores = []
                alists_tensorized = []

                # pair_all, score_all = zip(* list_all)
                # as_all, bs_all = zip(* pair_all)
                s= time.time()
                list_all_np = np.array(list_all)
                score_all = list_all_np[:,1].tolist()
                temp = np.array(list_all_np[:,0].tolist())
                as_all = temp[:,0].tolist()
                bs_all = temp[:,1].tolist()
                print(time.time()-s)
                for i in range(0, len(list_all), VAL_BATCH_SIZE):
                  batches.append(list_all[i:i+VAL_BATCH_SIZE])
                  alists.append(list(as_all[i:i+VAL_BATCH_SIZE]))
                  blists.append(list(bs_all[i:i+VAL_BATCH_SIZE]))
                  scores.append(torch.tensor(score_all[i:i+VAL_BATCH_SIZE]).cuda())
                  alists_tensorized.append(torch.tensor(list(as_all[i:i+VAL_BATCH_SIZE])))

                self.eval_batches[mode][metric]['batches'] = batches
                self.eval_batches[mode][metric]['alists'] = alists
                self.eval_batches[mode][metric]['blists'] = blists
                self.eval_batches[mode][metric]['scores'] = scores
                self.eval_batches[mode][metric]['alists_tensorized'] = alists_tensorized

    def create_batches(self,metric,mode):
      """
        create batches as is and return number of batches created
      """
      self.batches = self.eval_batches[mode][metric]['batches']
      self.alists = self.eval_batches[mode][metric]['alists']
      self.blists = self.eval_batches[mode][metric]['blists']
      self.scores = self.eval_batches[mode][metric]['scores']
      self.alists_tensorized = self.eval_batches[mode][metric]['alists_tensorized']
        
      self.num_batches = len(self.batches)  
      self.mode = mode

      return self.num_batches


    def fetch_batched_data_by_id_optimized(self,i):
        """             
        """
        assert(i < self.num_batches)  
        alist = self.alists[i]
        blist = self.blists[i]
        score = self.scores[i]
        if self.av.HASH_MODE=="dot":
            # query_tensors = torch.from_numpy(self.query_embeds_aug[self.mode])[alist]
            query_tensors = self.query_embeds_aug[self.mode][alist]
        elif self.av.HASH_MODE=="dot2":
            # query_tensors = torch.from_numpy(self.query_embeds_aug2[self.mode])[alist]
            query_tensors = self.query_embeds_aug2[self.mode][alist]
        else:
            # query_tensors = torch.from_numpy(self.query_embeds[self.mode])[alist]
            query_tensors = self.query_embeds[self.mode][alist]
        #query_set_sizes = self.query_set_sizes[alist]

        if self.av.HASH_MODE=="dot":
            # corpus_tensors = torch.from_numpy(self.corpus_embeds_aug)[blist]
            corpus_tensors = self.corpus_embeds_aug[blist]
        elif self.av.HASH_MODE=="dot2":
            # corpus_tensors = torch.from_numpy(self.corpus_embeds_aug2)[blist]
            corpus_tensors = self.corpus_embeds_aug2[blist]
        else:
            # corpus_tensors = torch.from_numpy(self.corpus_embeds)[blist]
            corpus_tensors = self.corpus_embeds[blist]
        #corpus_set_sizes = self.corpus_set_sizes[blist]
        target = score
        return corpus_tensors, query_tensors, target 

    def create_per_query_batches(self,mode):
        """
          create batches as is and return number of batches created
        """

        #whole_list = self.list_total_arranged_per_query[mode]
        #split_len  = self.corpus_embeds.shape[0]
        #self.batches = [whole_list[i:i + split_len] for i in range(0, len(whole_list), split_len)]
        self.alists = self.per_query_batches[mode]['alists']
        self.blists = self.per_query_batches[mode]['blists']
        self.scores = self.per_query_batches[mode]['scores']

        if mode=="train":
            self.alists,self.blists,self.scores = shuffle(self.alists,self.blists,self.scores)


        #for btch in self.batches:
        #    pair_all, score_all = zip(*btch)
        #    as_all, bs_all = zip(* pair_all)
        #    self.alists.append(list(as_all))
        #    self.blists.append(list(bs_all))
        #    self.scores.append(list(score_all))

     
        self.num_batches = len(self.alists)  
        self.mode = mode

        return self.num_batches

def evaluate_validation_loss_scloss(av,model, sampler, mode):
  model.eval()
  #npos = len(sampler.list_pos[mode])
  #nneg = len(sampler.list_neg[mode])

  total_loss = 0 
  total_bit_balance_loss = 0 
  total_ranking_loss = 0
  total_fence_sitting_loss = 0
  n_batches = sampler.create_per_query_batches(mode)
  for i in tqdm.tqdm(range(n_batches)):
    batch_corpus_tensors, batch_query_tensors, batch_target = sampler.fetch_batched_data_by_id_optimized(i)
    #batch_tensors = sampler.fetch_fmap_batched_data_by_id(i)
    loss,bit_balance_loss,ranking_loss, fence_sitting_loss = model.computeLoss(batch_corpus_tensors, batch_query_tensors, batch_target)
    total_loss = total_loss+loss.item()
    total_bit_balance_loss += bit_balance_loss.item() 
    total_ranking_loss += ranking_loss.item()
    total_fence_sitting_loss += fence_sitting_loss.item()

  return total_loss, total_bit_balance_loss, total_ranking_loss, total_fence_sitting_loss 


def evaluate_validation_loss(av,model, sampler, mode):
  model.eval()
  #npos = len(sampler.list_pos[mode])
  #nneg = len(sampler.list_neg[mode])

  total_loss = 0 
  total_bit_balance_loss = 0 
  total_decorrelation_loss = 0
  total_fence_sitting_loss = 0
  n_batches = sampler.create_fmap_batches(mode)
  for i in tqdm.tqdm(range(n_batches)):
    batch_tensors = sampler.fetch_fmap_batched_data_by_id(i)
    loss,bit_balance_loss,decorrelation_loss, fence_sitting_loss = model.computeLoss(batch_tensors, None, None)
    total_loss = total_loss+loss.item()
    total_bit_balance_loss += bit_balance_loss.item() 
    total_decorrelation_loss += decorrelation_loss.item()
    total_fence_sitting_loss += fence_sitting_loss.item()

  return total_loss, total_bit_balance_loss, total_decorrelation_loss, total_fence_sitting_loss 

def evaluate_embeddings_similarity(av,model, sampler, mode):
  model.eval()
  npos = len(sampler.list_pos[mode])
  nneg = len(sampler.list_neg[mode])

  pred = []
  sign_pred = []
  tan_pred = []

  n_batches = sampler.create_batches("ap",mode)
  for i in tqdm.tqdm(range(n_batches)):
    #ignoring target values and qids here since not needed for AP ranking score 
    batch_corpus_tensors,  batch_query_tensors, _ = sampler.fetch_batched_data_by_id_optimized(i)
    #if av.SIGN_EVAL:
    #corpus_hashcodes = torch.sign(model.forward(batch_corpus_tensors).data)
    #query_hashcodes  = torch.sign(model.forward(batch_query_tensors).data)
    #else:
    corpus_hashcodes = model.forward(batch_corpus_tensors).data
    query_hashcodes  = model.forward(batch_query_tensors).data
    sign_corpus_hashcodes = torch.sign(corpus_hashcodes) 
    sign_query_hashcodes = torch.sign(query_hashcodes)
    tan_corpus_hashcodes = torch.nn.Tanh()(av.TANH_TEMP*corpus_hashcodes) 
    tan_query_hashcodes  = torch.nn.Tanh()(av.TANH_TEMP*query_hashcodes)

    #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    prediction = (query_hashcodes * corpus_hashcodes).sum(-1)
    sign_prediction = (sign_query_hashcodes * sign_corpus_hashcodes).sum(-1)
    tan_prediction = (tan_query_hashcodes * tan_corpus_hashcodes).sum(-1)
    
    pred.append( prediction.data )
    sign_pred.append( sign_prediction.data )
    tan_pred.append( tan_prediction.data )

  all_pred = torch.cat(pred,dim=0) 
  all_sign_pred = torch.cat(sign_pred,dim=0) 
  all_tan_pred = torch.cat(tan_pred,dim=0) 
  labels = torch.cat((torch.ones(npos),torch.zeros(nneg)))
  ap_score = average_precision_score(labels.cpu(), all_pred.cpu())   
  sign_ap_score = average_precision_score(labels.cpu(), all_sign_pred.cpu())   
  tan_ap_score = average_precision_score(labels.cpu(), all_tan_pred.cpu())   
  
  # MAP computation
  all_ap = []
  all_sign_ap = []
  all_tan_ap = []
  pred = []
  sign_pred = []
  tan_pred = []
  n_batches = sampler.create_batches("map",mode)
  for i in tqdm.tqdm(range(n_batches)):
    #ignoring target values and qids here since not needed for AP ranking score 
    batch_corpus_tensors, batch_query_tensors, _ = sampler.fetch_batched_data_by_id_optimized(i)
    #if av.SIGN_EVAL:
    #corpus_hashcodes = torch.sign(model.forward(batch_corpus_tensors).data)
    #query_hashcodes  = torch.sign(model.forward(batch_query_tensors).data)
    #else:
    corpus_hashcodes = model.forward(batch_corpus_tensors).data
    query_hashcodes  = model.forward(batch_query_tensors).data
    sign_corpus_hashcodes = torch.sign(corpus_hashcodes) 
    sign_query_hashcodes = torch.sign(query_hashcodes)
    tan_corpus_hashcodes = torch.nn.Tanh()(av.TANH_TEMP*corpus_hashcodes) 
    tan_query_hashcodes  = torch.nn.Tanh()(av.TANH_TEMP*query_hashcodes)

    
    #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    prediction = (query_hashcodes * corpus_hashcodes).sum(-1)
    sign_prediction = (sign_query_hashcodes * sign_corpus_hashcodes).sum(-1)
    tan_prediction = (tan_query_hashcodes * tan_corpus_hashcodes).sum(-1)

    pred.append( prediction.data )
    sign_pred.append( sign_prediction.data )
    tan_pred.append( tan_prediction.data )

  all_pred = torch.cat(pred,dim=0)
  all_sign_pred = torch.cat(sign_pred,dim=0) 
  all_tan_pred = torch.cat(tan_pred,dim=0) 
  labels = sampler.labels_total_arranged_per_query[mode]
  corpus_size = sampler.corpus_embeds.shape[0]
  
  for q_id in tqdm.tqdm(range(sampler.query_embeds[mode].shape[0])):
    q_pred = all_pred[q_id * corpus_size : (q_id+1) * corpus_size]
    q_sign_pred = all_sign_pred[q_id * corpus_size : (q_id+1) * corpus_size]
    q_tan_pred = all_tan_pred[q_id * corpus_size : (q_id+1) * corpus_size]
    q_labels = labels[q_id * corpus_size : (q_id+1) * corpus_size]
    ap = average_precision_score(q_labels, q_pred.cpu())
    sign_ap = average_precision_score(q_labels, q_sign_pred.cpu())
    tan_ap = average_precision_score(q_labels, q_tan_pred.cpu())
    all_ap.append(ap)
    all_sign_ap.append(sign_ap)
    all_tan_ap.append(tan_ap)
  return ap_score, all_ap, np.mean(all_ap),\
         sign_ap_score, all_sign_ap, np.mean(all_sign_ap) ,\
         tan_ap_score, all_tan_ap, np.mean(all_tan_ap) 



def set_sigmoid_args(av):
    if av.DATASET_NAME == "msweb294":
        av.sigmoid_a = -3.3263
        av.sigmoid_b = 3.7716
    elif av.DATASET_NAME == "msnbc294":
        av.sigmoid_a = -3.2829
        av.sigmoid_b = 3.7897   
    elif av.DATASET_NAME == "msnbc294_1"or av.DATASET_NAME == "msnbc294_2":
        av.sigmoid_a = -2.3229
        av.sigmoid_b =  3.9496    
    elif av.DATASET_NAME == "msnbc294_3":
        av.sigmoid_a = -2.1684088706970215 
        av.sigmoid_b = 3.986318588256836     
    elif av.DATASET_NAME == "msnbc294_4":
        av.sigmoid_a =  -2.07367205619812
        av.sigmoid_b =  3.9828364849090576   
    elif av.DATASET_NAME == "msnbc294_5":
        av.sigmoid_a =  -2.0759339332580566
        av.sigmoid_b =  3.9921586513519287    
    elif av.DATASET_NAME == "msnbc294_6":
        av.sigmoid_a = -2.060725688934326
        av.sigmoid_b = 3.9899673461914062     
    elif av.DATASET_NAME == "msnbc294_7":
        av.sigmoid_a = -1.982824444770813
        av.sigmoid_b = 4.00326681137085     
    elif av.DATASET_NAME == "msweb294_1":
        av.sigmoid_a = -3.3088
        av.sigmoid_b =  3.7709    
    elif av.DATASET_NAME == "syn":
        av.sigmoid_a = -1
        av.sigmoid_b = 4     
    else:
        raise NotImplementedError()
        
def run_hashcode_gen(av):
    if(av.DATASET_NAME=="msnbc294"):
        av.sigmoid_a = -3.2829
        av.sigmoid_b = 3.7897

    set_sigmoid_args(av)

    pickle_fp = av.DIR_PATH + "/data/hashcodePickles/"+av.DESC+"_hashcode_mat.pkl"
    if not os.path.exists(pickle_fp):
        device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
        train_data = FmapData(av)
        model = HashCodeTrainer(av).to(device)
        cnt = 0
        for param in model.parameters():
            cnt=cnt+torch.numel(param)
        logger.info("no. of params in model: %s",cnt)
        es = EarlyStoppingModule(av,av.ES)
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=av.LEARNING_RATE,
                                    weight_decay=av.WEIGHT_DECAY)
      
        run = 0
        while not av.NOT_RUN_TILL_ES or run < av.NUM_RUNS:
            start_time = time.time()
            if av.LOSS_TYPE == "permgnn_loss_hingeemb":
                n_batches = train_data.create_fmap_batches(mode="train")
            elif av.LOSS_TYPE == "sc_loss_hingeemb":
                n_batches = train_data.create_per_query_batches(mode="train")
            else:
                n_batches = train_data.create_batches_with_p2n(mode="train")
            epoch_loss =0
            epoch_bit_balance_loss = 0 
            epoch_decorrelation_loss = 0
            epoch_fence_sitting_loss = 0
            epoch_ranking_loss = 0

             
            for i in tqdm.tqdm(range(n_batches)):
                optimizer.zero_grad()
                if av.LOSS_TYPE == "permgnn_loss_hingeemb":
                    batch_tensors = train_data.fetch_fmap_batched_data_by_id(i)

                    loss,bit_balance_loss,decorrelation_loss, fence_sitting_loss  = model.computeLoss(batch_tensors, None,None)
                    epoch_bit_balance_loss += bit_balance_loss.item() 
                    epoch_decorrelation_loss += decorrelation_loss.item()
                    epoch_fence_sitting_loss += fence_sitting_loss.item()
                elif av.LOSS_TYPE == "sc_loss_hingeemb" or av.LOSS_TYPE == "flora_hingeemb" or av.LOSS_TYPE == "flora_hingeemb2":
                    batch_corpus_tensors, batch_query_tensors, batch_target = train_data.fetch_batched_data_by_id_optimized(i)

                    loss,bit_balance_loss, fence_sitting_loss, ranking_loss  = model.computeLoss(batch_corpus_tensors, batch_query_tensors, batch_target)
                    epoch_bit_balance_loss += bit_balance_loss.item() 
                    epoch_fence_sitting_loss += fence_sitting_loss.item()
                    epoch_ranking_loss += ranking_loss.item()
                else:    
                    batch_corpus_tensors, batch_query_tensors, batch_target = train_data.fetch_batched_data_by_id_optimized(i)

                    #TODO: Figure out issues with loading data tensor to GPU   
                    loss = model.computeLoss(batch_corpus_tensors, batch_query_tensors, batch_target)
                

                #totalLoss = (av.LAMBDA1/num_nodes)*loss1 + (av.LAMBDA2/num_nodes)*loss2 + ((1-(av.LAMBDA1+av.LAMBDA2))/(num_nodes**2))*loss3
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()   
                end_time = time.time()
            if av.LOSS_TYPE == "permgnn_loss_hingeemb":
                logger.info("Epoch: %d loss: %f bit_balance_loss: %f decorrelation_loss: %f fence_sitting_loss: %f time: %.2f", run,epoch_loss,epoch_bit_balance_loss,epoch_decorrelation_loss, epoch_fence_sitting_loss, end_time-start_time)
            elif av.LOSS_TYPE == "sc_loss_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb" or av.LOSS_TYPE == "flora_hingeemb2":
                logger.info("Epoch: %d loss: %f bit_balance_loss: %f ranking_loss: %f fence_sitting_loss: %f time: %.2f", run,epoch_loss,epoch_bit_balance_loss,epoch_ranking_loss, epoch_fence_sitting_loss, time.time()-start_time)
            else:
                logger.info("Epoch: %d loss: %f time: %.2f", run,epoch_loss, end_time-start_time)
            
            start_time = time.time()
            if av.LOSS_TYPE != "permgnn_loss_hingeemb" and av.LOSS_TYPE != "sc_loss_hingeemb" and av.LOSS_TYPE != "flora_hingeemb" and av.LOSS_TYPE != "flora_hingeemb2":
                tr_ap_score,tr_all_ap,tr_map_score, tr_sign_ap_score,tr_all_sign_ap,tr_sign_map_score, tr_tan_ap_score,tr_all_tan_ap,tr_tan_map_score  = evaluate_embeddings_similarity(av,model,train_data,mode="train")
                if av.NO_TANH:
                    logger.info("Run: %d TRAIN ap_score: %.6f map_score: %.6f sign_ap_score: %.6f sign_map_score: %.6f tan_ap_score: %.6f tan_map_score: %.6f Time: %.2f",run,tr_ap_score,tr_map_score,tr_sign_ap_score,tr_sign_map_score,tr_tan_ap_score,tr_tan_map_score,time.time()-start_time)
                else:
                    logger.info("Run: %d TRAIN ap_score: NA map_score: NA sign_ap_score: %.6f sign_map_score: %.6f tan_ap_score: %.6f tan_map_score: %.6f Time: %.2f",run,tr_sign_ap_score,tr_sign_map_score,tr_tan_ap_score,tr_tan_map_score,time.time()-start_time)

            start_time = time.time()
            if av.LOSS_TYPE != "permgnn_loss_hingeemb" and av.LOSS_TYPE != "sc_loss_hingeemb"  and av.LOSS_TYPE != "flora_hingeemb"  and av.LOSS_TYPE != "flora_hingeemb2":
                ap_score,all_ap,map_score, sign_ap_score,all_sign_ap,sign_map_score, tan_ap_score,all_tan_ap,tan_map_score = evaluate_embeddings_similarity(av,model,train_data, mode="val")
                if av.NO_TANH:
                    logger.info("Run: %d VAL ap_score: %.6f map_score: %.6f sign_ap_score: %.6f sign_map_score: %.6f tan_ap_score: %.6f tan_map_score: %.6f Time: %.2f",run,ap_score,map_score,sign_ap_score,sign_map_score,tan_ap_score,tan_map_score,time.time()-start_time)
                else:
                    logger.info("Run: %d VAL ap_score: NA map_score: NA sign_ap_score: %.6f sign_map_score: %.6f tan_ap_score: %.6f tan_map_score: %.6f Time: %.2f",run,sign_ap_score,sign_map_score,tan_ap_score,tan_map_score,time.time()-start_time)
            elif  av.LOSS_TYPE == "sc_loss_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb2":
                val_loss,total_bit_balance_loss,total_ranking_loss, total_fence_sitting_loss = evaluate_validation_loss_scloss(av,model,train_data, mode="val")
                logger.info("Epoch: %d VAL loss: %f bit_balance_loss: %f ranking_loss: %f fence_sitting_loss: %f time: %.2f", run,val_loss,total_bit_balance_loss,total_ranking_loss, total_fence_sitting_loss, time.time()-start_time)
            else:
                val_loss,total_bit_balance_loss,total_decorrelation_loss, total_fence_sitting_loss = evaluate_validation_loss(av,model,train_data, mode="val")
                logger.info("Epoch: %d VAL loss: %f bit_balance_loss: %f decorrelation_loss: %f fence_sitting_loss: %f time: %.2f", run,val_loss,total_bit_balance_loss,total_decorrelation_loss, total_fence_sitting_loss, end_time-start_time)
                #logger.info("Epoch: %d VAL loss: %f time: %.2f", run,val_loss, time.time()-start_time)

            if not av.NOT_RUN_TILL_ES:
                if av.LOSS_TYPE == "permgnn_loss_hingeemb" or av.LOSS_TYPE == "sc_loss_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb2":
                    if es.check([-val_loss],model,run):
                        break
                else:    
                    if av.SIGN_EVAL:
                        if es.check([sign_map_score],model,run):
                            break
                    else:
                        assert av.NO_TANH , print("Please use sign eval with tanh hashcode")
                        if es.check([map_score],model,run):
                            break
            run+=1
       
        #generate and dump hashcode  pickles
        #IMP: Load best validation model here
        checkpoint = es.load_best_model()
        model.load_state_dict(checkpoint['model_state_dict'])      

        all_hashcodes = {}
        corpus_hashcodes = torch.zeros((train_data.corpus_embeds.shape[0], av.hcode_dim))
        bsz = 40000
        for i in tqdm.tqdm(range(0, train_data.corpus_embeds.shape[0],bsz)):
            if av.HASH_MODE=="dot":
                # corpus_hashcodes[i:i+bsz,:] = model.forward(torch.from_numpy(train_data.corpus_embeds_aug[i:i+bsz,:])).data
                corpus_hashcodes[i:i+bsz,:] = model.forward(train_data.corpus_embeds_aug[i:i+bsz,:]).data
            elif av.HASH_MODE=="dot2":
                # corpus_hashcodes[i:i+bsz,:] = model.forward(torch.from_numpy(train_data.corpus_embeds_aug2[i:i+bsz,:])).data
                corpus_hashcodes[i:i+bsz,:] = model.forward(train_data.corpus_embeds_aug2[i:i+bsz,:]).data
            else:
                # corpus_hashcodes[i:i+bsz,:] = model.forward(torch.from_numpy(train_data.corpus_embeds[i:i+bsz,:])).data
                corpus_hashcodes[i:i+bsz,:] = model.forward(train_data.corpus_embeds[i:i+bsz,:]).data

        query_hashcodes = {}
        for mode in ["train", "val"]:
            if av.HASH_MODE=="dot":
                # query_hashcodes[mode] =  model.forward(torch.from_numpy(train_data.query_embeds_aug[mode])).data
                query_hashcodes[mode] =  model.forward(train_data.query_embeds_aug[mode]).data
            elif av.HASH_MODE=="dot2":
                # query_hashcodes[mode] =  model.forward(torch.from_numpy(train_data.query_embeds_aug2[mode])).data
                query_hashcodes[mode] =  model.forward(train_data.query_embeds_aug2[mode]).data
            else:
                # query_hashcodes[mode] =  model.forward(torch.from_numpy(train_data.query_embeds[mode])).data
                query_hashcodes[mode] =  model.forward(train_data.query_embeds[mode]).data

        all_hashcodes['query'] = query_hashcodes
        all_hashcodes['corpus'] = corpus_hashcodes
        logger.info("Dumping trained hashcode pickle at %s",pickle_fp)
        with open(pickle_fp, 'wb') as f:
            pickle.dump(all_hashcodes, f)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                 type=str,   default="logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",               type=bool,  default=True)
  ap.add_argument("--has_cuda",                type=bool,  default=torch.cuda.is_available())
  ap.add_argument("--ES",                      type=int,   default=5)      
  #ap.add_argument("--RUN_TILL_ES",                    type=bool,  default=True)
  ap.add_argument("--NUM_RUNS",                       type=int,   default=10)
  ap.add_argument("--delta",                   type=int,   default=15)
  ap.add_argument("--num_q",                   type=int,   default=200)
  ap.add_argument("--num_c_perq",              type=int,   default=100)
  ap.add_argument("--num_cpos_perq",           type=int,   default=10)
  ap.add_argument("--num_cneg_perq",           type=int,   default=90)
  ap.add_argument("--embed_dim",               type=int,   default=20)
  #ap.add_argument("--num_c_perq",              type=int,   default=1000)
  #ap.add_argument("--num_cpos_perq",           type=int,   default=100)
  #ap.add_argument("--num_cneg_perq",           type=int,   default=900)
  #ap.add_argument("--embed_dim",               type=int,   default=10)
  ap.add_argument("--WEIGHT_DECAY",            type=float, default=5*10**-4)
  ap.add_argument("--LEARNING_RATE",           type=float, default=1e-5)    
  ap.add_argument("--P2N",                     type=float, default=1.0)    
  ap.add_argument("--MARGIN",                  type=float, default=1.0)    
  ap.add_argument("--SCLOSS_MARGIN",           type=float, default=1.0)    
  ap.add_argument("--TANH_TEMP",               type=float, default=1.0)    
  ap.add_argument("--FENCE_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--DECORR_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--C1_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--WEAKSUP_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--m_load",                  type=int,   default=10000)
  ap.add_argument("--m_use",                   type=int,   default=1000)
  ap.add_argument("--BATCH_SIZE",              type=int,   default=1024)
  ap.add_argument("--a",                       type=int,   default=-100)
  ap.add_argument("--b",                       type=int,   default=100)
  ap.add_argument("--hcode_dim",               type=int,   default=16)
  ap.add_argument("--flora_dim",               type=int,   default=10)
  ap.add_argument("--num_hash_tables",         type=int,   default=10)
  ap.add_argument("--subset_size",             type=int,   default=8)
  ap.add_argument("--sc_subset_size",             type=int,   default=8)
  ap.add_argument("--synthT",                  type=int,   default=38) 
  #ap.add_argument("--synthT",                  type=int,   default=133) 
  ap.add_argument("--SCALE",                   type=int,   default=1) 
  ap.add_argument("--sigmoid_a",               type=float,   default=-3.3263) 
  ap.add_argument("--sigmoid_b",               type=float,   default=3.7716) 
  ap.add_argument("--T",                       type=float,   default=37) 
  ap.add_argument("--T1",                      type=float,   default=0.1)
  #ap.add_argument("--T1",                      type=int,   default=1)
  ap.add_argument("--K",                       type=int,   default=50)
  ap.add_argument("--DIR_PATH",                type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",            type=str,   default="syn", help="syn/msnbc/msweb/graphs")
  ap.add_argument("--TASK",                    type=str,   default="hinge", help="cos/hinge/dot/dotperquery")
  ap.add_argument("--LOSS_TYPE",               type=str,   default="cos_ap", help="cos_ap/dot_ce")
  ap.add_argument("--CsubQ",                          action='store_true')
  ap.add_argument("--HASH_MODE",               type=str,   default="fhash", help="fhash/cosine/dot")
  ap.add_argument("--SPECIAL_FHASH",           type=str,   default="", help="")
  ap.add_argument("--pickle_fp",               type=str,   default="",help="path/to/datasets")
  ap.add_argument("--trained_cosine_fmap_pickle_fp",  type=str,   default="",help="path/to/datasets")
  ap.add_argument("--FMAP_SELECT",           type=int,   default=-1, help="")
  ap.add_argument("--SPLIT",                   type=str,   default="train", help="train/val/test/all")
  ap.add_argument("--DEBUG",                   action='store_true')
  ap.add_argument("--NOT_RUN_TILL_ES",          action='store_true')
  ap.add_argument("--NO_TANH",                  action='store_true')
  ap.add_argument("--SIGN_EVAL",                  action='store_true')
  ap.add_argument("--DOT_HASH",                  action='store_true')
  ap.add_argument("--INIT_GAUSS",                  action='store_true')
  ap.add_argument("--use_pretrained_fmap",                  action='store_true')
  ap.add_argument("--SilverSig",                  action='store_true')
  ap.add_argument("--use_pretrained_hcode",                  action='store_true')
  ap.add_argument("--HIDDEN_LAYERS",               type=int, nargs='*',  default=[])
  ap.add_argument("--LAYER_NORMALIZE",                action='store_true')
  ap.add_argument("--DESC",           type=str,   default="", help="")

  av = ap.parse_args()


  HID_ARCH = "".join([f"RL_{dim}_" for dim in av.HIDDEN_LAYERS])
  av.IN_ARCH = "L" +  HID_ARCH + ("Tanh" if not av.NO_TANH else "") + ("Lnorm" if av.LAYER_NORMALIZE else "")+\
          ("InitGauss" if av.INIT_GAUSS else "Init_KH")
  av.c_sub = "CsubQ" if av.CsubQ else ""
  silver = "SilverSig" if av.SilverSig else ""
  dothash = "DotHash" if av.HASH_MODE=="dot" else ""
  dot2hash = "Dot2Hash" if av.HASH_MODE=="dot2" else ""
  flora = "Flora" if av.HASH_MODE=="flora" else ""
  sc_loss_subset = str(av.sc_subset_size) if (av.LOSS_TYPE=="sc_loss_hingeemb" and av.sc_subset_size!=8) else ""
  
  av.DESC= av.TASK+dothash+dot2hash+silver+flora+av.c_sub+"_"+av.DATASET_NAME+ "_MARGIN" + str(av.MARGIN) +"_muse" + str(av.m_use) + "_T" + str(av.T) + "_Scale" + str(av.SCALE) + "_hcode" + str(av.hcode_dim)+ "_fmapSelect_" + str(av.FMAP_SELECT)+ "_SpecialFhash_" + (av.SPECIAL_FHASH if av.SPECIAL_FHASH!="" else "Asym")+  "_LOSS_" + av.LOSS_TYPE+ sc_loss_subset + (("_tanh_temp"+str(av.TANH_TEMP)) if not av.NO_TANH  else "")+(("_fence_" + str(av.FENCE_LAMBDA)) if av.FENCE_LAMBDA!=0.0 else "")+(("_decorr_" + str(av.DECORR_LAMBDA)) if av.DECORR_LAMBDA!=0.0 else "")+(("_C1loss_" + str(av.C1_LAMBDA)) if av.C1_LAMBDA!=0.0 else "")+(("_SClossMargin" + str(av.SCLOSS_MARGIN)) if av.SCLOSS_MARGIN!=1.0 else "")  +(("_weaksup_" + str(av.WEAKSUP_LAMBDA)) if av.WEAKSUP_LAMBDA!=0.0 else "")+"_arch_"+ av.IN_ARCH  +("SignEval" if av.SIGN_EVAL else "NoSignEval") 
  av.logpath = av.logpath+"_"+av.DESC+datetime.now().isoformat()
  set_log(av)
  logger.info("Command line")
  logger.info('\n'.join(sys.argv[:]))

  #if av.want_cuda and av.has_cuda:
  #  torch.cuda.set_device(av.CUDA)
  #logger.info(f"using cuda: {torch.cuda.current_device()}")
    
  # Set random seeds
  seed = 4
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)
  torch.backends.cudnn.deterministic = False
  #  torch.backends.cudnn.benchmark = True
  # torch.autograd.set_detect_anomaly(True)
  run_hashcode_gen(av)


#python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --MARGIN=1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" 
#python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --MARGIN=1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash"  --LOSS_TYPE="dot_ce"
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --MARGIN=1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 20000 --TASK="hinge" --HASH_MODE="fhash" --ES=50 
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=100 --m_load=1000 --MARGIN=1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 2000 1000 500 200    --TASK="hinge" --HASH_MODE="fhash" --ES=50 
#CUDA_VISIBLE_DEVICES=3 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=10 --m_load=100 --MARGIN=1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 200 100     --TASK="hinge" --HASH_MODE="fhash" --ES=50 
#CUDA_VISIBLE_DEVICES=4 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=100 --m_load=1000 --MARGIN=1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 2000 1000 500 200    --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="dot_ce" --NO_TANH 
#CUDA_VISIBLE_DEVICES=3 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=10 --m_load=100 --MARGIN=1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 200 100     --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ce" --NO_TANH 
############################################
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.5 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH 
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.5 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 128 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 128 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.5 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 128 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=3 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 256 128 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=3 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 256 128 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=4 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.5 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS 256 128 64 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  
#
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="cos_ap" --NO_TANH  --INIT_GAUSS

#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="tan_dot_ap" --NO_TANH  
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=1.0 
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=10 
#CUDA_VISIBLE_DEVICES=3 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=0.1 
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=10


#best bet yet 
#CUDA_VISIBLE_DEVICES=4 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=100

#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=100
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.5 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=100
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=1 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=100

#CUDA_VISIBLE_DEVICES=3 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=100 --FENCE_LAMBDA=0.2
#CUDA_VISIBLE_DEVICES=4 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=100 --FENCE_LAMBDA=0.5
#CUDA_VISIBLE_DEVICES=4 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000 --MARGIN=0.05 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="dot_ap" --SIGN_EVAL --TANH_TEMP=100 --FENCE_LAMBDA=0.8


#loooong file  : logfile_hinge_msweb_MARGIN0.1_muse1000_T3_Scale1_hcode64_LOSS_cos_ap_arch_LTanh2022-08-27T23:56:19.037312


#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=100 --FENCE_LAMBDA=0.3 --DECORR_LAMBDA=0.3
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=100 --FENCE_LAMBDA=0.25 --DECORR_LAMBDA=0.25 --WEAKSUP_LAMBDA=0.25


#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=10 --FENCE_LAMBDA=0.25 --DECORR_LAMBDA=0.25 --WEAKSUP_LAMBDA=0.25
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.25 --DECORR_LAMBDA=0.25 --WEAKSUP_LAMBDA=0.25
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0.3
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.3 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0.1
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.3 --DECORR_LAMBDA=0.1 --WEAKSUP_LAMBDA=0.3
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.3 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0.3



#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=10 --SPECIAL_FHASH="SymQuery"
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=10 --SPECIAL_FHASH="SymCorpus"
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=10 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH="SymQuery"
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH="SymCorpus"
#CUDA_VISIBLE_DEVICES=2 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#
#
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=10 --SPECIAL_FHASH="SymQuery"
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438  --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=10 --SPECIAL_FHASH="SymCorpus"
#CUDA_VISIBLE_DEVICES=3 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438  --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=10 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438  --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH="SymQuery"
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438  --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH="SymCorpus"
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438  --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --LOSS_TYPE="permgnn_loss_hingeemb" --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA=0.3 --WEAKSUP_LAMBDA=0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""




#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.25 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.05 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.1 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.15 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.2 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.3 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.35 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.4 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.45 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msweb"  --m_use=1 --m_load=10 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.5 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
# sighinge best model - sigmoid_a : -3.2829, sigmoid_b : 3.7897
# CUDA_VISIBLE_DEVICES=0 python -m src.hashcode_trainer_hingeemb --DATASET_NAME="msnbc294"  --m_use=10 --m_load=2940 --T=146.53720092773438 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50 --LOSS_TYPE="sc_loss_hingeemb" --TANH_TEMP=1.0 --FENCE_LAMBDA=0.1 --C1_LAMBDA=0.5 --trained_cosine_fmap_pickle_fp="" --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
