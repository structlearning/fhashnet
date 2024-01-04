import multiprocessing
import torch
#torch.backends.cuda.matmul.allow_tf32 = False
import pickle
from common import logger, set_log
import networkx as nx
import random 
import numpy as np
from src.utils import cudavar
import torch.nn.functional as F
import os
from sklearn.metrics import average_precision_score
import time
import itertools
from src.earlystopping import EarlyStoppingModule
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
import tqdm
import scipy
from random import randrange
from pebble import ProcessPool
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math

from src.hashing_main import * 
# from src.locality_sensitive_hashing_trained import LSH_tr
from src.locality_sensitive_hashing import *
from src.hashing_utils import *

def set_seed():
    # Set random seeds
  seed = 4
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)
  torch.backends.cudnn.deterministic = False

                                           

    


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  #ap.add_argument("--T",                       type=float,   default=37) 
  ap.add_argument("--TASK",                    type=str,   default="sighinge") 
  ap.add_argument("--DATASET_NAME",            type=str,   default="msnbc294_3") 
  ap.add_argument("--HASH_MODE",               type=str,   default="wmh") 
  #ap.add_argument("--m_use",                   type=int,   default=10) 
  ap.add_argument("--CORRUPT_GT",             action='store_true')

  av = ap.parse_args()

  SENTINEL = None 


  #NOTE
  av.T = 3.0
  #av.T = 38.0
  av.synthT = 38
  av.delta = 15
  av.num_q = 200
  av.num_cpos_perq = 10
  av.num_cneg_perq = 90
  av.T1=0.1
  av.m_use=10
  av.CsubQ = False
  av.pickle_fp = ""
  av.num_hash_tables=10
  av.want_cuda     = False
  av.has_cuda      = torch.cuda.is_available()
  av.a = -100
  av.b = 100
  av.SCALE = 1
  av.MARGIN=1.0
  av.MAX_QUERIES=100
  av.NUM_QUERIES=500
  av.LEARNING_RATE=1e-3
  av.LOSS="relu"
  av.BATCH_SIZE=1024
  av.P2N=1.0
  av.DEEPSET_VER="UA"
  av.ES=50
  av.EmbedType="Bert768"
  av.TEST=True
  av.DESC =""
  av.IN_HIDDEN_LAYERS = [294]
  av.IN_RELU = True
  av.OUT_HIDDEN_LAYERS = []
  av.OUT_RELU = False
  av.IN_D = 50
  av.PER_QUERY = False
  av.LAYER_NORMALIZE= False
  
  #av.DATASET_NAME="msnbc294"
  #av.m_use = 10
  #NOTE
  av.embed_dim = 294
  #av.embed_dim = 20
  av.m_load = av.m_use * av.embed_dim
  #av.TASK="sighinge"
  #av.HASH_MODE = "fhash"
  av.hcode_dim=64 
  av.FMAP_SELECT = -1
  av.SPECIAL_FHASH = ""
  av.SPLIT = "test"
  av.num_cid_remove=0
  av.use_pretrained_fmap = False
  av.use_pretrained_hcode = False
  av.trained_cosine_fmap_pickle_fp = ""
  av.DEBUG = False
  av.K = -1
  av.sigmoid_a = -3.2829
  av.sigmoid_b = 3.7897
  set_sigmoid_args(av)
  M=0
  av.subset_size = 8
  av.WMH = "chum"


  dat = pickle.load(open(f"./data/{av.DATASET_NAME}_embeds_{av.TASK}.pkl","rb"))
  cemb = dat['all_c']
  row_map = {}
  curr_idx = 0
  for idx in range(len(cemb)):
      if not all(cemb[idx]==0):
          row_map[curr_idx] = idx
          curr_idx = curr_idx+1

  def adjust_for_wmh_gt(op):
      for qidx in range(len(op[0]['asymhash'])):
          op[0]['asymhash'][qidx]=op[0]['asymhash'][qidx][:2]+\
                                  ([row_map[idx] for idx in op[0]['asymhash'][qidx][2]],)+\
                                  op[0]['asymhash'][qidx][3:]
      return op

  def inner_foo(av,wmh_type,d):
    av.WMH = wmh_type
    tmp_list = []

    for subset_size in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30]:
        av.subset_size = subset_size
        hash_op  = my_run_lsh(av)       
        op = preprocess_compute_all_scores_dec10(adjust_for_wmh_gt(hash_op),nohash_op)
        tmp_list.append(compute_all_scores_dec10(op))
    
    d.put((wmh_type, tmp_list))
    d.put(SENTINEL)



  queue = multiprocessing.Queue()
  procs = []
 
  all_notrain_wmh_sighinge_fhash = {}

  algo_list = ["minhash", "gollapudi2", "icws", "pcws", "ccws", "i2cws", "chum","licws"] 
  nohash_op = my_run_nohash(av)
  
  for ver in algo_list:   
    p =  multiprocessing.Process(target=inner_foo, args=(av,ver,queue))
    procs.append(p)
    p.start()

  seen_sentinel_count = 0
  while seen_sentinel_count < len(algo_list):
      a = queue.get()
      if a is SENTINEL:
          seen_sentinel_count += 1
      else:
          all_notrain_wmh_sighinge_fhash[a[0]] = a[1]

  for p in procs: 
      p.join()

  
  str_new  = "_Corrupt" if av.CORRUPT_GT else ""
  fp= f"{av.DATASET_NAME}_{av.TASK}_untrained_{av.HASH_MODE}{str_new}"
  pickle.dump(all_notrain_wmh_sighinge_fhash, open(fp,"wb"))
