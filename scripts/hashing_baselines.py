import multiprocessing
import os
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
    ap.add_argument("--HASH_MODE",               type=str,   default="cosine") 
    ap.add_argument("--DATASET_NAME",            type=str,   default="msnbc294_3") 
    ap.add_argument("--m_use",                   type=int,   default=10) 
    ap.add_argument("--sc_subset_size",          type=int,   default=8) 
    ap.add_argument("--fmap_mode",               type=str,   default="BCE3") 
    ap.add_argument("--LOSS_TYPE",               type=str,   default="sc_loss_hingeemb") 
    ap.add_argument("--CORRUPT_GT",             action='store_true')

    av = ap.parse_args()



    SENTINEL = None 
    
    #HINGE EMBED + TRAINED FHASH + SCLOSS HASHCODE
    s = time.time()
    av.has_cuda                   = torch.cuda.is_available()
    av.want_cuda = False
    av.DIR_PATH                   ="."
    av.CsubQ = False
    av.a = -100
    av.b = 100
    #NOTE
    av.T = 3.0
    #av.T = 38.0
    av.synthT = 38
    av.delta = 15
    av.num_q = 200
    av.num_cpos_perq = 10
    av.num_cneg_perq = 90
    av.T1=0.1
    av.num_hash_tables = 10
    av.subset_size = 8
    av.SPLIT = "test"
    av.K=-1
    av.DEBUG = False
    # av.MARGIN=1.0
    av.SCALE=1
    av.hcode_dim=64
    #av.LOSS_TYPE = "sc_loss"
    av.TANH_TEMP = 1.0
    av.trained_cosine_fmap_pickle_fp = ""
    av.pickle_fp = ""
    av.FMAP_SELECT=-1
    av.FENCE_LAMBDA=0.1
    av.WEAKSUP_LAMBDA=0
    av.DECORR_LAMBDA = 0
    av.C1_LAMBDA = 0
    #NOTE
    #av.HASH_MODE = "dot"
    #av.TASK = "sighinge"
    av.HIDDEN_LAYERS = []
    av.FMAP_HIDDEN_LAYERS = []
    av.DESC = ""
    av.use_pretrained_fmap = False
    av.use_pretrained_hcode = True
    av.NO_TANH = False
    av.sigmoid_a = -3.3263
    av.sigmoid_b =  3.7716
    set_sigmoid_args(av)
    av.SilverSig = False
    av.E2EVER=""
    av.USE_FMAP_BCE = False
    av.USE_FMAP_BCE2 = False
    av.USE_FMAP_BCE3 = False
    av.USE_FMAP_MSE = False
    av.USE_FMAP_PQR = False
    if av.fmap_mode == "BCE3":
        av.USE_FMAP_BCE3 = True
    elif av.fmap_mode == "BCE":
        av.USE_FMAP_BCE = True
    elif av.fmap_mode == "MSE":
        av.USE_FMAP_MSE = True
    
    all_metric_list2_dec10_parallel = []
    
    variations = [(0.05, 10,"Asym","AsymFmapCos")]
    
    av.SCLOSS_MARGIN=1.0
    
    
    c1_val = [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
           0.6 , 0.65, 0.7 , 0.8 , 0.85]
    #c1_val = [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
    #       0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85]
    # c1_val = [0.4 , 0.45, 0.5 , 0.55,
    #        0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85]
    
    # c1_val = [0.05]
    #NOTE
    str_new  = "_Corrupt" if av.CORRUPT_GT else ""
    if av.HASH_MODE == "cosine":
        fp = f"{av.DATASET_NAME}_{av.TASK}_EnemyCos_{av.LOSS_TYPE}{av.sc_subset_size}{str_new}.pkl"
    elif av.HASH_MODE == "dot":
        fp = f"{av.DATASET_NAME}_{av.TASK}_EnemyDot_{av.LOSS_TYPE}{av.sc_subset_size}{str_new}.pkl"
    elif av.HASH_MODE == "flora":
        fp = f"{av.DATASET_NAME}_{av.TASK}_EnemyFlora_{av.LOSS_TYPE}{av.sc_subset_size}{str_new}.pkl"
    else:
        raise NotImplementedError()

    av.flora_dim=10
    #av.sc_subset_size=8
    av.ES = 50
    av.INIT_GAUSS = False
    av.LAYER_NORMALIZE = False
    #NOTE
    av.embed_dim = 294
    #av.embed_dim = 20
    #av.m_use = 10
    av.m_load = av.embed_dim*av.m_use 
    av.SIGN_EVAL = False
    torch.set_num_threads(1) 
    
    def inner_foo(av,dval,d):
        if av.LOSS_TYPE == "sc_loss_hingeemb" or av.LOSS_TYPE == "flora_hingeemb" or av.LOSS_TYPE == "flora_hingeemb2":
            av.C1_LAMBDA=dval
        else:
            av.DECORR_LAMBDA=dval

        #all_hash_op13_1[dval] = {}
        #temp_dict = {}
        tmp_list = []
        for subset_size in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            av.subset_size = subset_size   
            hash_op = my_run_lsh(av)
            op = preprocess_compute_all_scores_dec10(hash_op,nohash_op)
            tmp_list.append(compute_all_scores_dec10(op))
            #param_list2_dec10.append((margin,v1,v2,v3,dval))
            #temp_dict[subset_size] = op
            
        d.put((dval, tmp_list))
        d.put(SENTINEL)
        return
    
    
    queue = multiprocessing.Queue()
    procs = []
    all_dict = {} 
    all_metric_list2_dec10_parallel = []
    
    for margin,v1,v2,v3 in variations:
        keystr= f"{margin}_{v1}_{v2}_{v3}"
        #all_hash_op2_all[keystr] = {}
        av.MARGIN=1.0 #margin
        av.tr_fmap_dim=v1
        av.SPECIAL_FHASH = v2
        av.FMAP_LOSS_TYPE = v3
        if av.LOSS_TYPE == "sc_loss_hingeemb" or av.LOSS_TYPE == "flora_hingeemb" or av.LOSS_TYPE == "flora_hingeemb2" :
            av.C1_LAMBDA=0.05
        else:
            av.DECORR_LAMBDA=0.05
        nohash_op = my_run_nohash(av)
        for dval in c1_val:
            
            p =  multiprocessing.Process(target=inner_foo, args=(av,dval,queue))
            procs.append(p)
            p.start()
            
        seen_sentinel_count = 0
        while seen_sentinel_count < len(c1_val):
            a = queue.get()
            if a is SENTINEL:
                seen_sentinel_count += 1
            else:
                all_dict [a[0]] = a[1]
            
        
    
        for p in procs: 
            p.join()
    
        for dval in c1_val:
            all_metric_list2_dec10_parallel.extend(all_dict[dval])
            
    pickle.dump(all_metric_list2_dec10_parallel, open(fp,"wb"))    
    print(time.time()-s)



