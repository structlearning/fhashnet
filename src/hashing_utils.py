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


def set_seed():
    # Set random seeds
  seed = 4
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)
  torch.backends.cudnn.deterministic = False


def set_sigmoid_args(av):
    if av.DATASET_NAME == "msweb294":
        av.sigmoid_a = -3.3263
        av.sigmoid_b = 3.7716
    elif av.DATASET_NAME == "msnbc294":
        av.sigmoid_a = -3.2829
        av.sigmoid_b = 3.7897   
    elif av.DATASET_NAME == "msnbc294_1" or av.DATASET_NAME == "msnbc294_2":
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
        

def my_run_lsh(av,num_cid_remove=0):
    set_sigmoid_args(av)
    corpus_embeds = fetch_corpus_embeddings(av)
    num_corpus = corpus_embeds.shape[0]

    query_embeds = fetch_query_embeddings(av)
    ground_truth = fetch_ground_truths(av)

    new_corpus_embeds, new_ground_truth = elim_excess_cids(av,corpus_embeds, ground_truth, num_cid_remove)
    if av.CORRUPT_GT:
        noise_frac = 0.1
        new_ground_truth = {}
        for qid in range(len(query_embeds)):
            npos = len(ground_truth[qid])
            n_corrput = int(np.ceil(noise_frac*npos))
        
            #positive items to remove
            sc = asym_sim(av,query_embeds[qid],corpus_embeds[ground_truth[qid]])
            remove_cids = list(np.array(ground_truth[qid])[np.argsort(sc)[::-1][-n_corrput:]])
            #negative items to add
            neg_cemb = np.delete(corpus_embeds,ground_truth[qid],axis=0 )
            neg_cids = np.delete(np.arange(corpus_embeds.shape[0]),ground_truth[qid] )
            neg_sc = asym_sim(av,query_embeds[qid],neg_cemb)
            add_cids = list(neg_cids[np.argsort(neg_sc)[::-1][:n_corrput]])
        
            new_ground_truth[qid] = list(set(ground_truth[qid]) - set(remove_cids)) + add_cids
        ground_truth = new_ground_truth
  
    num_qitems = query_embeds.shape[0]
    assert(len(ground_truth.keys()) == num_qitems)

    set_seed()
    #This will init the k hash functions, each of dimension d
    lsh = LSH(av)
    #This will generate feature maps and index corpus items
    lsh.index_corpus(new_corpus_embeds,av.HASH_MODE)

    if av.HASH_MODE == "flora":
        curr_task = av.TASK
        av.TASK = "sighinge"
        cembeds = fetch_corpus_embeddings(av)
        qembeds = fetch_query_embeddings(av)
        lsh.set_flora_info(qembeds, cembeds)

        #lsh.cembeds = cembeds
        #lsh.qembeds = qembeds
        av.TASK=curr_task
        

    all_hashing_info_dict = {}
    all_hashing_info_dict['asymhash']= []
#     all_hashing_info_dict['nohash']= []
    
    for qid,qemb in enumerate(query_embeds): 
        #reshape qemb to 1*d
        qemb_reshaped = np.expand_dims(qemb, axis=0)
        all_hashing_info_dict['asymhash'].append(lsh.retrieve(qemb_reshaped,av.K, hash_mode=av.HASH_MODE, no_bucket=False,qid=qid))


  
    # =====================Time Analysis================================
    time_logger_dict = {}
    other_data_dict = {}
    for k in all_hashing_info_dict.keys(): 
        time_logger_dict[k] = {}
        other_data_dict[k] = {}
        for k1 in all_hashing_info_dict[k][0][3].keys(): 
            time_logger_dict[k][k1] = {}
            for k2 in all_hashing_info_dict[k][0][3][k1].keys():
                time_logger_dict[k][k1][k2] = 0
                for qidx in range(len(query_embeds)):
                    try:
                        time_logger_dict[k][k1][k2] += all_hashing_info_dict[k][qidx][3][k1][k2]
                    except:
                        print(k2)
                        print(all_hashing_info_dict[k][qidx][3][k1])
                        exit(1)
        for k1 in all_hashing_info_dict[k][0][4].keys():
            other_data_dict[k][k1] = 0
            for qidx in range(len(query_embeds)):
                other_data_dict[k][k1] += all_hashing_info_dict[k][qidx][4][k1]
    
    return  all_hashing_info_dict,time_logger_dict


def my_run_nohash(av,num_cid_remove=0):
    set_sigmoid_args(av)    
    corpus_embeds = fetch_corpus_embeddings(av)
    num_corpus = corpus_embeds.shape[0]

    query_embeds = fetch_query_embeddings(av)
    ground_truth = fetch_ground_truths(av)

    new_corpus_embeds, new_ground_truth = elim_excess_cids(av,corpus_embeds, ground_truth, num_cid_remove)
    if av.CORRUPT_GT:
        noise_frac = 0.1
        new_ground_truth = {}
        for qid in range(len(query_embeds)):
            npos = len(ground_truth[qid])
            n_corrput = int(np.ceil(noise_frac*npos))
        
            #positive items to remove
            sc = asym_sim(av,query_embeds[qid],corpus_embeds[ground_truth[qid]])
            remove_cids = list(np.array(ground_truth[qid])[np.argsort(sc)[::-1][-n_corrput:]])
            #negative items to add
            neg_cemb = np.delete(corpus_embeds,ground_truth[qid],axis=0 )
            neg_cids = np.delete(np.arange(corpus_embeds.shape[0]),ground_truth[qid] )
            neg_sc = asym_sim(av,query_embeds[qid],neg_cemb)
            add_cids = list(neg_cids[np.argsort(neg_sc)[::-1][:n_corrput]])
        
            new_ground_truth[qid] = list(set(ground_truth[qid]) - set(remove_cids)) + add_cids
        ground_truth = new_ground_truth
  
    num_qitems = query_embeds.shape[0]
    assert(len(ground_truth.keys()) == num_qitems)

    set_seed()
    #This will init the k hash functions, each of dimension d
    lsh = LSH(av)
    #This will generate feature maps and index corpus items
    lsh.index_corpus(new_corpus_embeds,av.HASH_MODE)

    if av.HASH_MODE == "flora":
        curr_task = av.TASK
        av.TASK = "sighinge"
        cembeds = fetch_corpus_embeddings(av)
        qembeds = fetch_query_embeddings(av)
        lsh.set_flora_info(qembeds, cembeds)

        #lsh.cembeds = cembeds
        #lsh.qembeds = qembeds
        av.TASK=curr_task

    all_hashing_info_dict = {}
    all_hashing_info_dict['nohash']= []
#     all_hashing_info_dict['nohash']= []
    
    for qid,qemb in enumerate(query_embeds): 
        #reshape qemb to 1*d
        qemb_reshaped = np.expand_dims(qemb, axis=0)
        all_hashing_info_dict['nohash'].append(lsh.retrieve(qemb_reshaped,av.K, hash_mode=av.HASH_MODE, no_bucket=True,qid=qid))


  
    # =====================Time Analysis================================
    time_logger_dict = {}
    other_data_dict = {}
    for k in all_hashing_info_dict.keys(): 
        time_logger_dict[k] = {}
        other_data_dict[k] = {}
        for k1 in all_hashing_info_dict[k][0][3].keys(): 
            time_logger_dict[k][k1] = {}
            for k2 in all_hashing_info_dict[k][0][3][k1].keys():
                time_logger_dict[k][k1][k2] = 0
                for qidx in range(len(query_embeds)):
                    try:
                        time_logger_dict[k][k1][k2] += all_hashing_info_dict[k][qidx][3][k1][k2]
                    except:
                        print(k2)
                        print(all_hashing_info_dict[k][qidx][3][k1])
                        exit(1)
        for k1 in all_hashing_info_dict[k][0][4].keys():
            other_data_dict[k][k1] = 0
            for qidx in range(len(query_embeds)):
                other_data_dict[k][k1] += all_hashing_info_dict[k][qidx][4][k1]
    
    return  all_hashing_info_dict,time_logger_dict,new_ground_truth



from collections import Counter


def custom_ap_at_K(ground_truth, pred_cids, pred_scores, K, len_gt = None):
    """
        ground_truth : set of relevant corpus ids
        pred_cids : list of predicted corpus ids
        pred_scores: list of predicted scores for the pred_cids
        K : required top K items (only needed to check and throw exception)
    """
#     if K>=0:
#         try:
#             assert len(pred_cids)==K
#         except Exception as e:
#             logger.exception(e)
#             logger.info(f"# ground truth={len(ground_truth)}, # preds={len(pred_cids)}")
    assert (K>0)    
    sorted_pred_scores = sorted(((e, i) for i, e in enumerate(pred_scores)), reverse=True)[:K]
    sum_precision= 0 
    positive_count = 0
    position_count = 0
    for sc, idx in sorted_pred_scores:
        position_count += 1
        #check if label=1
        if pred_cids[idx] in ground_truth: 
             positive_count +=1 
             sum_precision += (positive_count/position_count)
    if positive_count ==0: 
        average_precision = 0
    else: 
        average_precision = sum_precision/positive_count
#     if len_gt is not None: 
#         average_precision = sum_precision/len_gt
#     else:
#         average_precision = sum_precision/len(ground_truth)
    return average_precision

def compute_topK_score_dec2( hash_scores, K, M=0):
    """
    """
    if len(hash_scores)==0:
        #TODO: Discuss
        total_hash_score = 0# min(nohash_scores)
    else:
        total_hash_score= np.sum(np.array(hash_scores[:K]) + M)


    return total_hash_score

def compute_topK_score_norm_dec2( hash_scores, K, M=0):
    """
    """
    if len(hash_scores)==0:
        #TODO: Discuss
        total_hash_score = 0# min(nohash_scores)
    else:
        total_hash_score= np.sum(np.array(hash_scores[:K]) + M)/len(hash_scores)


    return total_hash_score


def preprocess_compute_all_scores_dec10(op_hash,op_nohash,Q_gt_lens=None,thresh=None):
    all_hashing_info_dict = {}
    all_hashing_info_dict['asymhash'] = op_hash[0]['asymhash']
    all_hashing_info_dict['nohash'] = op_nohash[0]['nohash']
        
    time_logger_dict = {}
    time_logger_dict['asymhash'] = op_hash[1]['asymhash']
    time_logger_dict['nohash'] = op_nohash[1]['nohash']
    
    new_ground_truth = op_nohash[2]
    num_corpus = op_nohash[0]['nohash'][0][0]
 
    all_topK_score_dec2_10 = []
    all_topK_score_norm_dec2_10 = []
    all_topK_score_dec2_20 = []
    all_topK_score_norm_dec2_20 = []
    all_topK_score_dec2_50 = []
    all_topK_score_norm_dec2_50 = []
    all_topK_score_dec2_200 = []
    all_topK_score_norm_dec2_200 = []
    all_topK_score_dec2_500 = []
    all_topK_score_norm_dec2_500 = []
    
    num_evals = []
    all_customap_hash = []
    all_customap_atk_hash = []

    
    num_qitems = len(op_hash[0]['asymhash'])

    for qidx in range(num_qitems):
        all_topK_score_dec2_10.append(compute_topK_score_dec2(all_hashing_info_dict['asymhash'][qidx][1], 10))
        all_topK_score_norm_dec2_10.append(compute_topK_score_norm_dec2(all_hashing_info_dict['asymhash'][qidx][1], 10))
        all_topK_score_dec2_20.append(compute_topK_score_dec2(all_hashing_info_dict['asymhash'][qidx][1], 20))
        all_topK_score_norm_dec2_20.append(compute_topK_score_norm_dec2(all_hashing_info_dict['asymhash'][qidx][1], 20))
        all_topK_score_dec2_50.append(compute_topK_score_dec2(all_hashing_info_dict['asymhash'][qidx][1], 50))
        all_topK_score_norm_dec2_50.append(compute_topK_score_norm_dec2(all_hashing_info_dict['asymhash'][qidx][1], 50))
        all_topK_score_dec2_200.append(compute_topK_score_dec2(all_hashing_info_dict['asymhash'][qidx][1], 200))
        all_topK_score_norm_dec2_200.append(compute_topK_score_norm_dec2(all_hashing_info_dict['asymhash'][qidx][1], 200))
        all_topK_score_dec2_500.append(compute_topK_score_dec2(all_hashing_info_dict['asymhash'][qidx][1], 500))
        all_topK_score_norm_dec2_500.append(compute_topK_score_norm_dec2(all_hashing_info_dict['asymhash'][qidx][1], 500))
        num_evals.append(all_hashing_info_dict['asymhash'][qidx][0])
        all_customap_hash.append(custom_ap(set(new_ground_truth[qidx]),all_hashing_info_dict['asymhash'][qidx][2],all_hashing_info_dict['asymhash'][qidx][1],-1, len(set(new_ground_truth[qidx]))))
        all_customap_atk_hash.append(custom_ap_at_K(set(new_ground_truth[qidx]),all_hashing_info_dict['asymhash'][qidx][2],all_hashing_info_dict['asymhash'][qidx][1],50, len(set(new_ground_truth[qidx]))))

    if Q_gt_lens is not None:
        all_customap_hash = np.array(all_customap_hash) [Q_gt_lens<thresh]
        all_topK_score_dec2_10 = np.array(all_topK_score_dec2_10) [Q_gt_lens<thresh]
 


    return all_topK_score_dec2_10, all_topK_score_norm_dec2_10,time_logger_dict,num_evals,\
            np.mean(all_customap_hash),all_topK_score_dec2_20, all_topK_score_norm_dec2_20,\
            all_topK_score_dec2_50, all_topK_score_norm_dec2_50,\
            all_topK_score_dec2_200, all_topK_score_norm_dec2_200,\
            all_topK_score_dec2_500, all_topK_score_norm_dec2_500,\
            np.mean(all_customap_atk_hash)
                                           
def preprocess_compute_all_scores_exhaustive_dec10(op_hash,op_nohash,Q_gt_lens=None,thresh=None):
    all_hashing_info_dict = {}
    all_hashing_info_dict['asymhash'] = op_hash[0]['asymhash']
    all_hashing_info_dict['nohash'] = op_nohash[0]['nohash']
        
    time_logger_dict = {}
    time_logger_dict['asymhash'] = op_hash[1]['asymhash']
    time_logger_dict['nohash'] = op_nohash[1]['nohash']
    
    new_ground_truth = op_nohash[2]
    num_corpus = op_nohash[0]['nohash'][0][0]
 
    all_topK_score_dec2_10 = []
    all_topK_score_norm_dec2_10 = []
    all_topK_score_dec2_20 = []
    all_topK_score_norm_dec2_20 = []
    all_topK_score_dec2_50 = []
    all_topK_score_norm_dec2_50 = []
    all_topK_score_dec2_200 = []
    all_topK_score_norm_dec2_200 = []
    all_topK_score_dec2_500 = []
    all_topK_score_norm_dec2_500 = []
    num_evals = []
    all_customap_nohash = []
    all_customap_atk_nohash = []

    
    num_qitems = len(op_hash[0]['asymhash'])

    for qidx in range(num_qitems):
        all_topK_score_dec2_10.append(compute_topK_score_dec2(all_hashing_info_dict['nohash'][qidx][1], 10))
        all_topK_score_norm_dec2_10.append(compute_topK_score_norm_dec2(all_hashing_info_dict['nohash'][qidx][1], 10))
        all_topK_score_dec2_20.append(compute_topK_score_dec2(all_hashing_info_dict['nohash'][qidx][1], 20))
        all_topK_score_norm_dec2_20.append(compute_topK_score_norm_dec2(all_hashing_info_dict['nohash'][qidx][1], 20))
        all_topK_score_dec2_50.append(compute_topK_score_dec2(all_hashing_info_dict['nohash'][qidx][1], 50))
        all_topK_score_norm_dec2_50.append(compute_topK_score_norm_dec2(all_hashing_info_dict['nohash'][qidx][1], 50))
        all_topK_score_dec2_200.append(compute_topK_score_dec2(all_hashing_info_dict['nohash'][qidx][1], 200))
        all_topK_score_norm_dec2_200.append(compute_topK_score_norm_dec2(all_hashing_info_dict['nohash'][qidx][1], 200))
        all_topK_score_dec2_500.append(compute_topK_score_dec2(all_hashing_info_dict['nohash'][qidx][1], 500))
        all_topK_score_norm_dec2_500.append(compute_topK_score_norm_dec2(all_hashing_info_dict['nohash'][qidx][1], 500))
        num_evals.append(all_hashing_info_dict['nohash'][qidx][0])
        all_customap_nohash.append(custom_ap(set(new_ground_truth[qidx]),all_hashing_info_dict['nohash'][qidx][2],all_hashing_info_dict['nohash'][qidx][1],-1, len(set(new_ground_truth[qidx]))))
        all_customap_atk_nohash.append(custom_ap_at_K(set(new_ground_truth[qidx]),all_hashing_info_dict['nohash'][qidx][2],all_hashing_info_dict['nohash'][qidx][1],50, len(set(new_ground_truth[qidx]))))

    if Q_gt_lens is not None:
        all_customap_hash = np.array(all_customap_hash) [Q_gt_lens<thresh]
        all_topK_score_dec2_10 = np.array(all_topK_score_dec2_10) [Q_gt_lens<thresh]
      

    return all_topK_score_dec2_10, all_topK_score_norm_dec2_10,time_logger_dict,num_evals,\
            np.mean(all_customap_nohash),all_topK_score_dec2_20, all_topK_score_norm_dec2_20,\
            all_topK_score_dec2_50, all_topK_score_norm_dec2_50,\
            all_topK_score_dec2_200, all_topK_score_norm_dec2_200,\
            all_topK_score_dec2_500, all_topK_score_norm_dec2_500,\
            np.mean(all_customap_atk_nohash)



def compute_all_scores_dec10(op):
    total_time = sum(list(op[2]['asymhash']['real'].values())) - op[2]['asymhash']['real']['take_time']
 
    aggr_topKscore_10 = np.mean(np.array(op[0]))
    aggr_topKscore_norm_10 = np.mean(np.array(op[1]))
    num_evals = np.mean(op[3])
    aggr_topKscore_20 = np.mean(np.array(op[5]))
    aggr_topKscore_norm_20 = np.mean(np.array(op[6]))
    aggr_topKscore_50 = np.mean(np.array(op[7]))
    aggr_topKscore_norm_50 = np.mean(np.array(op[8]))
    aggr_topKscore_200 = np.mean(np.array(op[9]))
    aggr_topKscore_norm_200 = np.mean(np.array(op[10]))
    aggr_topKscore_500 = np.mean(np.array(op[11]))
    aggr_topKscore_norm_500 = np.mean(np.array(op[12]))

    return  aggr_topKscore_10, aggr_topKscore_norm_10, total_time, num_evals,\
            op[4],aggr_topKscore_20, aggr_topKscore_norm_20,\
            aggr_topKscore_50, aggr_topKscore_norm_50,\
            aggr_topKscore_200, aggr_topKscore_norm_200,\
            aggr_topKscore_500, aggr_topKscore_norm_500, op[13]
                                           
def compute_all_scores_exhaustive_dec10(op):
    total_time = sum(list(op[2]['nohash']['real'].values())) - op[2]['nohash']['real']['take_time']

    aggr_topKscore_10 = np.mean(np.array(op[0]))
    aggr_topKscore_norm_10 = np.mean(np.array(op[1]))
    num_evals = np.mean(op[3])        
    aggr_topKscore_20 = np.mean(np.array(op[5]))
    aggr_topKscore_norm_20 = np.mean(np.array(op[6]))
    aggr_topKscore_50 = np.mean(np.array(op[7]))
    aggr_topKscore_norm_50 = np.mean(np.array(op[8]))
    aggr_topKscore_200 = np.mean(np.array(op[9]))
    aggr_topKscore_norm_200 = np.mean(np.array(op[10]))
    aggr_topKscore_500 = np.mean(np.array(op[11]))
    aggr_topKscore_norm_500 = np.mean(np.array(op[12]))

    return  aggr_topKscore_10, aggr_topKscore_norm_10, total_time, num_evals,\
            op[4],aggr_topKscore_20, aggr_topKscore_norm_20,\
            aggr_topKscore_50, aggr_topKscore_norm_50,\
            aggr_topKscore_200, aggr_topKscore_norm_200,\
            aggr_topKscore_500, aggr_topKscore_norm_500, op[13]

