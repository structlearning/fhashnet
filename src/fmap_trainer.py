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
from src.locality_sensitive_hashing import *
#import matplotlib.pyplot as plt
#from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import tqdm
from datetime import datetime
from src.earlystopping import EarlyStoppingModule
import math

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

def pairwise_ranking_loss_similarity_per_query(predPos, predNeg, qidPos, qidNeg, av):
    
    assert qidPos.shape == predPos.shape and qidNeg.shape == predNeg.shape, f"qidPos.shape: {qidPos.shape}, predPos.shape: {predPos.shape}, qidNeg.shape: {qidNeg.shape}, predNeg.shape: {predNeg.shape}"
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_qid_1 = qidPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    expanded_qid_2 = qidNeg.unsqueeze(0).expand(n_1, n_2, dim)

    ell = av.MARGIN + expanded_2 - expanded_1
    loss = torch.nn.ReLU()(ell) * (expanded_qid_1 == expanded_qid_2)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(torch.sum(expanded_qid_1 == expanded_qid_2))#, torch.sum(expanded_qid_1 == expanded_qid_2) / (n_1 * n_2)

class FmapTrainer(nn.Module):
    """
      Fetch fmaps for q, c 
      feed into NN(LRL)
      Compute loss on final FMAP 
    """
    def __init__(self, av):
      super(FmapTrainer, self).__init__()
      self.av = av
      
      self.init_net = []
      if self.av.FMAP_SELECT == -1:
        self.inner_hs = [self.av.m_use * self.av.embed_dim * 4] + av.FMAP_HIDDEN_LAYERS + [self.av.tr_fmap_dim]
      else:
        self.inner_hs = [self.av.FMAP_SELECT] + av.FMAP_HIDDEN_LAYERS + [self.av.tr_fmap_dim]
      for h0, h1 in zip(self.inner_hs, self.inner_hs[1:]):
            lin = torch.nn.Linear(h0, h1)
            if self.av.INIT_GAUSS:
                nn.init.normal_(lin.weight)
            self.init_net.append(lin)
            if av.LAYER_NORMALIZE:
              self.init_net.append(torch.nn.LayerNorm(h1))
            self.init_net.append(torch.nn.ReLU())
      self.init_net.pop() # pop the last relu/tanh
      self.init_net = torch.nn.Sequential(*self.init_net)
      
      self.tanh  = nn.Tanh()
      self.bce_loss = torch.nn.BCEWithLogitsLoss() 
      self.bce_loss_with_prob = torch.nn.BCELoss()
      self.mse_loss = torch.nn.MSELoss()
      #nn.init.normal_(self.hash_linear1.weight)

    def forward(self, fmaps,isQ=True):
        """
            :param  Fmaps
            :return  Hcodes
        """
        code = self.init_net(cudavar(self.av,fmaps))
        return code/torch.norm(code,dim=-1,keepdim=True)


    def computeLoss(self, cfmaps, qfmaps, targets, batch_query_ids):
      """
        :param   cfmaps  : corpus fourier maps
        :param   qfmaps  : query fourier maps
        :param   targets : ground truth scores 0/1
        :return  loss   : Hinge ranking loss
      """
      q_maps = self.forward(qfmaps.cuda(),isQ=True)
      c_maps = self.forward(cfmaps.cuda(),isQ=False)
      preds = (q_maps*c_maps).sum(-1)
      
      if self.av.USE_FMAP_BCE:
        preds = (preds+1)/2
        loss = self.bce_loss(preds,targets.cuda())   
      elif self.av.USE_FMAP_BCE2:
        targets[targets==0]=-1
        loss = self.bce_loss(preds,targets.cuda())   
      elif self.av.USE_FMAP_BCE3:
        preds = (preds+1)/2
        loss = self.bce_loss_with_prob(preds,targets.cuda())   
      elif self.av.USE_FMAP_MSE:
        targets[targets==0]=-1
        loss = self.mse_loss(preds,targets.cuda())   
      elif self.av.USE_FMAP_PQR:
        predPos = preds[targets>0.5]
        predNeg = preds[targets<0.5]
        qidPos = batch_query_ids[targets>0.5].cuda()
        qidNeg = batch_query_ids[targets<0.5].cuda()
        loss  = pairwise_ranking_loss_similarity_per_query(predPos.unsqueeze(1),predNeg.unsqueeze(1),qidPos.unsqueeze(1),qidNeg.unsqueeze(1), self.av)
        #loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.av.MARGIN)
      else:
        predPos = preds[targets>0.5]
        predNeg = preds[targets<0.5]
        loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.av.MARGIN)

      return loss


class AsymFmapTrainer(nn.Module):
    """
      Fetch fmaps for q, c 
      feed into NN(LRL)
      Compute loss on final FMAP 
    """
    def __init__(self, av):
      super(AsymFmapTrainer, self).__init__()
      self.av = av
      
      self.init_net = []
      if self.av.FMAP_SELECT == -1:
        self.inner_hs = [self.av.m_use * self.av.embed_dim * 4] + av.FMAP_HIDDEN_LAYERS + [self.av.tr_fmap_dim]
      else:
        self.inner_hs = [self.av.FMAP_SELECT] + av.FMAP_HIDDEN_LAYERS + [self.av.tr_fmap_dim]
      for h0, h1 in zip(self.inner_hs, self.inner_hs[1:]):
            lin = torch.nn.Linear(h0, h1)
            if self.av.INIT_GAUSS:
                nn.init.normal_(lin.weight)
            self.init_net.append(lin)
            if av.LAYER_NORMALIZE:
              self.init_net.append(torch.nn.LayerNorm(h1))
            self.init_net.append(torch.nn.ReLU())
      self.init_net.pop() # pop the last relu/tanh

      self.init_net = torch.nn.Sequential(*self.init_net)
      
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


      self.tanh  = nn.Tanh()
      self.bce_loss = torch.nn.BCEWithLogitsLoss() 
      self.bce_loss_with_prob = torch.nn.BCELoss() 
      self.mse_loss = torch.nn.MSELoss()
      
      #nn.init.normal_(self.hash_linear1.weight)

    def forward(self, fmaps,isQ=True):
        """
            :param  Fmaps
            :return  Hcodes
        """
        if isQ:
            code = self.init_net(cudavar(self.av,fmaps))
        else:
            code = self.init_cnet(cudavar(self.av,fmaps))
        return code/torch.norm(code,dim=-1,keepdim=True)


    def computeLoss(self, cfmaps, qfmaps, targets, batch_query_ids):
      """
        :param   cfmaps  : corpus fourier maps
        :param   qfmaps  : query fourier maps
        :param   targets : ground truth scores 0/1
        :return  loss   : Hinge ranking loss
      """
      q_maps = self.forward(qfmaps.cuda(),isQ=True)
      c_maps = self.forward(cfmaps.cuda(),isQ=False)
      preds = (q_maps*c_maps).sum(-1)  
      
      if self.av.USE_FMAP_BCE:
        preds = (preds+1)/2
        loss = self.bce_loss(preds,targets.cuda())   
      elif self.av.USE_FMAP_BCE2:
        targets[targets==0]=-1
        loss = self.bce_loss(preds,targets.cuda())   
      elif self.av.USE_FMAP_BCE3:
        preds = (preds+1)/2
        loss = self.bce_loss_with_prob(preds,targets.cuda())   
      elif self.av.USE_FMAP_MSE:
        targets[targets==0]=-1
        loss = self.mse_loss(preds,targets.cuda())   
      elif self.av.USE_FMAP_PQR:
        predPos = preds[targets>0.5]
        predNeg = preds[targets<0.5]
        qidPos = batch_query_ids[targets>0.5].cuda()
        qidNeg = batch_query_ids[targets<0.5].cuda()
        loss  = pairwise_ranking_loss_similarity_per_query(predPos.unsqueeze(1),predNeg.unsqueeze(1),qidPos.unsqueeze(1),qidNeg.unsqueeze(1), self.av)
        #loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.av.MARGIN)
      else:
        predPos = preds[targets>0.5]
        predNeg = preds[targets<0.5]
        loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.av.MARGIN)

      return loss


class FmapData(object):
    def __init__(self, av): 
        self.av = av
        self.lsh = LSH(self.av)
        corpus_embeds_fetch_start = time.time()
        self.corpus_embeds = fetch_corpus_embeddings(self.av)
        corpus_embeds_fetch_time = time.time() - corpus_embeds_fetch_start
        logger.info(f"Corpus embeds shape: {self.corpus_embeds.shape}, time={corpus_embeds_fetch_time}")
        
        corpusfmaps_start_time = time.time()

        self.batch_sz = 40000
        fmaps_init = time.time()
        self.corpus_fmaps = cudavar(self.av,torch.zeros((self.corpus_embeds.shape[0], self.av.m_use * self.corpus_embeds.shape[1] * 4)))
        logger.info(f"init corpus fmaps, shape={self.corpus_fmaps.shape}, time={time.time()-fmaps_init}")
        for i in tqdm.tqdm(range(0, self.corpus_embeds.shape[0],self.batch_sz)):
            if self.av.SPECIAL_FHASH == "SymCorpus":
                self.corpus_fmaps[i:i+self.batch_sz,:] = torch.from_numpy(self.lsh.generate_fmap(self.av, self.corpus_embeds[i:i+self.batch_sz], isQuery=False)).type(torch.float)
            elif self.av.SPECIAL_FHASH == "SymQuery":
                self.corpus_fmaps[i:i+self.batch_sz,:] = torch.from_numpy(self.lsh.generate_fmap(self.av, self.corpus_embeds[i:i+self.batch_sz], isQuery=True)).type(torch.float)
            else:
                self.corpus_fmaps[i:i+self.batch_sz,:] = torch.from_numpy(self.lsh.generate_fmap(self.av, self.corpus_embeds[i:i+self.batch_sz], isQuery=False)).type(torch.float)
            
        if not av.FMAP_SELECT == -1:
            self.corpus_fmaps = self.corpus_fmaps[:,:self.av.FMAP_SELECT]
        
        corpusfmaps_time = time.time() - corpusfmaps_start_time
        logger.info(f"Corpus fmaps shape: {self.corpus_fmaps.shape}, time={corpusfmaps_time}")
        
        self.query_embeds  = {}
        self.query_fmaps  = {}
        self.ground_truth = {}
        self.list_pos = {}
        self.list_neg = {} 
        self.list_total_arranged_per_query = {}
        self.labels_total_arranged_per_query = {}
        self.eval_batches = {}
        #for mode in ["train", "test", "val"]:
        for mode in ["train", "val"]:
        
            self.av.SPLIT = mode
            self.query_embeds[mode] = fetch_query_embeddings(self.av)
        
            if self.av.SPECIAL_FHASH == "SymCorpus":
                self.query_fmaps[mode] = cudavar(self.av,torch.from_numpy(self.lsh.generate_fmap(self.av, self.query_embeds[mode], isQuery=False)).type(torch.float))
            elif self.av.SPECIAL_FHASH == "SymQuery":
                self.query_fmaps[mode] = cudavar(self.av,torch.from_numpy(self.lsh.generate_fmap(self.av, self.query_embeds[mode], isQuery=True)).type(torch.float))
            else:
                self.query_fmaps[mode] = cudavar(self.av,torch.from_numpy(self.lsh.generate_fmap(self.av, self.query_embeds[mode], isQuery=True)).type(torch.float))

            if not av.FMAP_SELECT == -1:
                self.query_fmaps[mode] = self.query_fmaps[mode][:,:self.av.FMAP_SELECT]

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

    def create_fmap_batches(self,mode):
        all_fmaps = torch.cat([self.query_fmaps[mode], self.corpus_fmaps])
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

        # pair_all, score_all = zip(* list_all)
        # as_all, bs_all = zip(* pair_all)
        list_all_np = np.array(list_all, dtype=object)
        score_all = list_all_np[:,1].tolist()
        temp = np.array(list_all_np[:,0].tolist())
        as_all = temp[:,0].tolist()
        bs_all = temp[:,1].tolist()

        for i in range(0, len(list_all), VAL_BATCH_SIZE):
          self.batches.append(list_all[i:i+VAL_BATCH_SIZE])
          self.alists.append(list(as_all[i:i+VAL_BATCH_SIZE]))
          self.blists.append(list(bs_all[i:i+VAL_BATCH_SIZE]))
          self.scores.append(list(score_all[i:i+VAL_BATCH_SIZE]))

     
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

      # lpos_pair, lposs = zip(*lpos)
      # lposa, lposb = zip(*lpos_pair)

      lpos_np = np.array(lpos, dtype=object)
      lposs = lpos_np[:,1].tolist()
      lpos_pair = np.array(lpos_np[:,0].tolist())
      lposa = lpos_pair[:,0].tolist()
      lposb = lpos_pair[:,1].tolist()

      # lneg_pair, lnegs = zip(*lneg)
      # lnega, lnegb = zip(*lneg_pair)

      lneg_np = np.array(lneg, dtype=object)
      lnegs = lneg_np[:,1].tolist()
      lneg_pair = np.array(lneg_np[:,0].tolist())
      lnega = lneg_pair[:,0].tolist()
      lnegb = lneg_pair[:,1].tolist()

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
      self.scores = [list(a+b) for (a,b) in zip(ss_pos[:self.num_batches],ss_neg[:self.num_batches])]
      self.alists_tensorized = [torch.tensor(list(a+b)) for (a,b) in zip(as_pos[:self.num_batches],as_neg[:self.num_batches])]
      self.mode = mode

      return self.num_batches

    def preprocess_create_batches(self,VAL_BATCH_SIZE=10000):
        #for mode in ["train", "test", "val"]:
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

                list_all_np = np.array(list_all, dtype=object)
                score_all = list_all_np[:,1].tolist()
                temp = np.array(list_all_np[:,0].tolist())
                as_all = temp[:,0].tolist()
                bs_all = temp[:,1].tolist()
                
                for i in range(0, len(list_all), VAL_BATCH_SIZE):
                  batches.append(list_all[i:i+VAL_BATCH_SIZE])
                  alists.append(list(as_all[i:i+VAL_BATCH_SIZE]))
                  blists.append(list(bs_all[i:i+VAL_BATCH_SIZE]))
                  scores.append(list(score_all[i:i+VAL_BATCH_SIZE]))
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
        query_tensors = self.query_fmaps[self.mode][alist]
        #query_set_sizes = self.query_set_sizes[alist]

        corpus_tensors = self.corpus_fmaps[blist]
        #corpus_set_sizes = self.corpus_set_sizes[blist]
        target = torch.tensor(score)
        return corpus_tensors, query_tensors, target, self.alists_tensorized[i] 

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
    batch_corpus_tensors,  batch_query_tensors, _, _ = sampler.fetch_batched_data_by_id_optimized(i)
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
    batch_corpus_tensors, batch_query_tensors, _, _ = sampler.fetch_batched_data_by_id_optimized(i)
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




def run_fmap_gen(av):
    pickle_fp = av.DIR_PATH + "/data/fmapPickles/"+av.DESC+"_fmap_mat.pkl"
    if not os.path.exists(pickle_fp):

        device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
        train_data = FmapData(av)
        if av.FMAP_LOSS_TYPE == "AsymFmapCos": 
            model = AsymFmapTrainer(av).to(device)
        elif av.FMAP_LOSS_TYPE == "FmapCos": 
            model = FmapTrainer(av).to(device)
        else:
            raise NotImplementedError()

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
            n_batches = train_data.create_batches_with_p2n(mode="train")
            epoch_loss =0

            start_time = time.time()
             
            for i in tqdm.tqdm(range(n_batches)):
                optimizer.zero_grad()
                batch_corpus_tensors, batch_query_tensors, batch_target, batch_query_ids = train_data.fetch_batched_data_by_id_optimized(i)
                #TODO: Figure out issues with loading data tensor to GPU   
                loss = model.computeLoss(batch_corpus_tensors, batch_query_tensors, batch_target, batch_query_ids)
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()   
            end_time = time.time()
            logger.info("Epoch: %d loss: %f time: %.2f", run,epoch_loss, end_time-start_time)
            
            start_time = time.time()
            tr_ap_score,tr_all_ap,tr_map_score, tr_sign_ap_score,tr_all_sign_ap,tr_sign_map_score, tr_tan_ap_score,tr_all_tan_ap,tr_tan_map_score  = evaluate_embeddings_similarity(av,model,train_data,mode="train")
            logger.info("Run: %d TRAIN ap_score: %.6f map_score: %.6f sign_ap_score: %.6f sign_map_score: %.6f tan_ap_score: %.6f tan_map_score: %.6f Time: %.2f",run,tr_ap_score,tr_map_score,tr_sign_ap_score,tr_sign_map_score,tr_tan_ap_score,tr_tan_map_score,time.time()-start_time)

            start_time = time.time()
            ap_score,all_ap,map_score, sign_ap_score,all_sign_ap,sign_map_score, tan_ap_score,all_tan_ap,tan_map_score = evaluate_embeddings_similarity(av,model,train_data, mode="val")
            logger.info("Run: %d VAL ap_score: %.6f map_score: %.6f sign_ap_score: %.6f sign_map_score: %.6f tan_ap_score: %.6f tan_map_score: %.6f Time: %.2f",run,ap_score,map_score,sign_ap_score,sign_map_score,tan_ap_score,tan_map_score,time.time()-start_time)
 
            if not av.NOT_RUN_TILL_ES:
                if es.check([map_score],model,run):
                    break
            run+=1
       
        #generate and dump fmap  pickles
        #IMP: Load best validation model here
        checkpoint = es.load_best_model()
        model.load_state_dict(checkpoint['model_state_dict'])      

        all_fmaps = {}
        corpus_fmaps = torch.zeros((train_data.corpus_embeds.shape[0], av.tr_fmap_dim))
        bsz = 40000
        for i in tqdm.tqdm(range(0, train_data.corpus_embeds.shape[0],bsz)):
            corpus_fmaps[i:i+bsz,:] = model.forward(train_data.corpus_fmaps[i:i+bsz,:],isQ=False).data
        query_fmaps = {}
        #for mode in ["train", "test", "val"]:
        for mode in ["train", "val"]:
            query_fmaps[mode] =  model.forward(train_data.query_fmaps[mode],isQ=True).data
        all_fmaps['query'] = query_fmaps
        all_fmaps['corpus'] = corpus_fmaps
        logger.info("Dumping trained fmap pickle at %s",pickle_fp)
        with open(pickle_fp, 'wb') as f:
            pickle.dump(all_fmaps, f)


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
  ap.add_argument("--TANH_TEMP",               type=float, default=1.0)    
  ap.add_argument("--FENCE_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--DECORR_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--WEAKSUP_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--m_load",                  type=int,   default=10000)
  ap.add_argument("--m_use",                   type=int,   default=1000)
  ap.add_argument("--BATCH_SIZE",              type=int,   default=1024)
  ap.add_argument("--a",                       type=int,   default=-100)
  ap.add_argument("--b",                       type=int,   default=100)
  ap.add_argument("--tr_fmap_dim",               type=int,   default=16)
  ap.add_argument("--hcode_dim",               type=int,   default=16)
  ap.add_argument("--num_hash_tables",         type=int,   default=10)
  ap.add_argument("--subset_size",             type=int,   default=8)
  ap.add_argument("--synthT",                  type=int,   default=38) 
  #ap.add_argument("--synthT",                  type=int,   default=133) 
  ap.add_argument("--SCALE",                   type=int,   default=1) 
  ap.add_argument("--T",                       type=float,   default=37) 
  ap.add_argument("--T1",                      type=float,   default=0.1)
  #ap.add_argument("--T1",                      type=int,   default=1)
  ap.add_argument("--K",                       type=int,   default=50)
  ap.add_argument("--DIR_PATH",                type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",            type=str,   default="syn", help="syn/msnbc/msweb/graphs")
  ap.add_argument("--TASK",                    type=str,   default="hinge", help="cos/hinge/dot/dotperquery")
  ap.add_argument("--FMAP_LOSS_TYPE",               type=str,   default="cos_ap", help="cos_ap/dot_ce")
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
  ap.add_argument("--SIGN_EVAL",                action='store_true')
  ap.add_argument("--INIT_GAUSS",               action='store_true')
  ap.add_argument("--USE_FMAP_BCE",             action='store_true')
  ap.add_argument("--USE_FMAP_BCE2",             action='store_true')
  ap.add_argument("--USE_FMAP_BCE3",             action='store_true')
  ap.add_argument("--USE_FMAP_MSE",             action='store_true')
  ap.add_argument("--USE_FMAP_PQR",  action='store_true')
  ap.add_argument("--FMAP_HIDDEN_LAYERS",       type=int, nargs='*',  default=[])
  ap.add_argument("--HIDDEN_LAYERS",            type=int, nargs='*',  default=[])
  ap.add_argument("--LAYER_NORMALIZE",          action='store_true')
  ap.add_argument("--use_pretrained_fmap",      action='store_true')
  ap.add_argument("--use_pretrained_hcode",    action='store_true')

  av = ap.parse_args()


  HID_ARCH = "".join([f"RL_{dim}_" for dim in av.FMAP_HIDDEN_LAYERS])
  av.IN_ARCH = "L" +  HID_ARCH #+ ("Tanh" if not av.NO_TANH else "") + ("Lnorm" if av.LAYER_NORMALIZE else "")+\
          #("InitGauss" if av.INIT_GAUSS else "Init_KH")
  av.c_sub = "CsubQ" if av.CsubQ else ""
  
  av.DESC= av.TASK+av.c_sub+"_"+av.DATASET_NAME+ "_MARGIN" + str(av.MARGIN) +"_muse" + str(av.m_use) + "_T" + str(av.T) + "_Scale" + str(av.SCALE) + "_TrFmapDim" + str(av.tr_fmap_dim)+ "_fmapSelect_" + str(av.FMAP_SELECT)+ "_SpecialFhash_" + (av.SPECIAL_FHASH if av.SPECIAL_FHASH!="" else "Asym")+  "_LOSS_" + av.FMAP_LOSS_TYPE +  "_arch_"+ av.IN_ARCH + ("_fmapBCE" if av.USE_FMAP_BCE else "") + ("_fmapBCE2" if av.USE_FMAP_BCE2 else "") + ("_fmapBCE3" if av.USE_FMAP_BCE3 else "") + ("_fmapMSE" if av.USE_FMAP_MSE else "")+ ("_fmapPQR" if av.USE_FMAP_PQR else "")
  #(("_tanh_temp"+str(av.TANH_TEMP)) if not av.NO_TANH  else "")+(("_fence_" + str(av.FENCE_LAMBDA)) if av.FENCE_LAMBDA!=0.0 else "")+(("_decorr_" + str(av.DECORR_LAMBDA)) if av.DECORR_LAMBDA!=0.0 else "") +(("_weaksup_" + str(av.WEAKSUP_LAMBDA)) if av.WEAKSUP_LAMBDA!=0.0 else "")+
 # +("SignEval" if av.SIGN_EVAL else "NoSignEval") 
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

  run_fmap_gen(av)



#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438 --SCALE=1 --tr_fmap_dim=16 --HIDDEN_LAYERS 16 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="AsymFmapCos" --MARGIN=1.0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438 --SCALE=1 --tr_fmap_dim=16 --HIDDEN_LAYERS 16 --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="FmapCos" --MARGIN=1.0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438 --SCALE=1 --tr_fmap_dim=16 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="AsymFmapCos" --MARGIN=1.0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1 --m_load=10  --T=146.53720092773438 --SCALE=1 --tr_fmap_dim=16 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="FmapCos" --MARGIN=1.0 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""



#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --tr_fmap_dim=10 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="FmapCos" --MARGIN=0.05 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --tr_fmap_dim=64 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="FmapCos" --MARGIN=0.05 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --tr_fmap_dim=10 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="AsymFmapCos" --MARGIN=0.05 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH=""
#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --tr_fmap_dim=10 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="FmapCos" --MARGIN=0.05 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH="SymCorpus"
#CUDA_VISIBLE_DEVICES=1 python -m src.fmap_trainer --DATASET_NAME="msweb" --m_use=1000 --m_load=10000  --T=3 --SCALE=1 --tr_fmap_dim=10 --HIDDEN_LAYERS --TASK="hinge" --HASH_MODE="fhash" --ES=50  --FMAP_LOSS_TYPE="FmapCos" --MARGIN=0.05 --pickle_fp="" --FMAP_SELECT=-1 --SPECIAL_FHASH="SymQuery"

