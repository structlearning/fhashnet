import argparse
import random
import time
import pickle
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from common import logger, set_log
from src.utils import cudavar
from src.locality_sensitive_hashing import *
from src.synthetic_data_generator import generate_labels
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import tqdm
#import falconn

np.set_printoptions(threshold=sys.maxsize)

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

def dot_sim(a, b):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the dot similarity a[i]*b[j] for all i and j.
    :return: Matrix with res[i][j]  = \sum(a[i]*b[j])
    """
    return (a[:,None,:]*b[None,:,:]).sum(-1)


def hinge_sim(a, b):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the asym hinge similarity -max(0,a[i]- b[j]) for all i and j.
    :return: Matrix with res[i][j]  = -max(0,a[i]- b[j])
    """
    return -(np.maximum((a[:,None,:]-b[None,:,:]),0)).sum(-1)

def normalized_cosine_similarity(a,b):
    """
    a,b numpy arrays of dim: m x d, and n x d
    output: m x n matrix, pairwise scores
    """
    return a @ b.T

def custom_ap(ground_truth, pred_cids, pred_scores, K, len_gt = None):
    """
        ground_truth : set of relevant corpus ids
        pred_cids : list of predicted corpus ids
        pred_scores: list of predicted scores for the pred_cids
        K : required top K items (only needed to check and throw exception)
    """
    if K>=0:
        try:
            assert len(pred_cids)==K
        except Exception as e:
            logger.exception(e)
            logger.info(f"# ground truth={len(ground_truth)}, # preds={len(pred_cids)}")
        
    sorted_pred_scores = sorted(((e, i) for i, e in enumerate(pred_scores)), reverse=True)
    sum_precision= 0 
    positive_count = 0
    position_count = 0
    for sc, idx in sorted_pred_scores:
        position_count += 1
        #check if label=1
        if pred_cids[idx] in ground_truth: 
             positive_count +=1 
             sum_precision += (positive_count/position_count)
    if len_gt is not None: 
        average_precision = sum_precision/len_gt
    else:
        average_precision = sum_precision/len(ground_truth)
    return average_precision


def precision_at_k(ground_truth, predictions, K):
    """
        ground_truth : set of corpus ids deemed relevant to any given query
        predictions  : set of corpus ids returned by retrieval model in top-K search
    """
    #TODO: decide what happens if assert fails
    if K>=0:
        try:
            assert len(predictions)==K
        except Exception as e:
            logger.exception(e)
            logger.info(f"# ground truth={len(ground_truth)}, # preds={len(predictions)}")
    tp = len(ground_truth.intersection(predictions))

    return 1.0*tp/K
    
def test_fourier_map(av):
    """
        1a. Fetch corupus embeddings
        1b. Fetch fourier map of corpus
        2a. Fetch query embeddings
        2b. Fetch fourier map of query
        3a. Compute asym_sim on (query, corpus) embeddings
        3b. Compute empirical mean using the samples
    """
    corpus_embeds = fetch_corpus_embeddings(av)
    assign_threshold(av, corpus_embeds)
    ws, pdfs = fetch_samples(av)
    ground_truth = fetch_ground_truths(av)
    corpus_fmaps = generate_fourier_map(av, corpus_embeds, ws, pdfs)
    logger.info(f"Corpus fmaps shape: {corpus_fmaps.shape}")
    query_embeds = fetch_query_embeddings(av)
    query_fmaps = generate_fourier_map(av, query_embeds, ws, pdfs, isQuery=True)
    logger.info(f"Query fmaps shape: {query_fmaps.shape}")
    num_q = query_embeds.shape[0]
    num_c = corpus_embeds.shape[0]
    
    # check if norms are constant
    v = np.linalg.norm(corpus_fmaps[0])
    assert np.all(np.isclose(v, np.linalg.norm(corpus_fmaps, axis=1)))
    logger.info("Successfully verified norm consistency of corpus fmaps")
    assert np.all(np.isclose(v, np.linalg.norm(query_fmaps, axis=1)))
    logger.info("Successfully verified norm consistency of query fmaps")
    for i in range(num_c):
        assert np.isclose(np.linalg.norm(corpus_fmaps[i]), v), f"v={v}, corpus={np.linalg.norm(corpus_fmaps[i])}"
    for i in range(num_q):
        assert np.isclose(np.linalg.norm(query_fmaps[i]), v), f"v={v}, query={np.linalg.norm(query_fmaps[i])}"

    sim = [None for _ in range(num_q)]
    empirical_mean = [None for _ in range(num_q)]
    for qidx in range(num_q):
        logger.info("#"*18 + f" qidx={qidx} " + "#"*18)
        sim[qidx] = asym_sim(av, query_embeds[qidx], corpus_embeds)
        # logger.info(f"sim.shape = {sim[qidx].shape}")
        # compare[qidx] = np.column_stack((sim[qidx], empirical_mean[qidx]))
        # logger.info(f"qidx={qidx}, compare={compare[qidx]}")
        # logger.info(f"qidx={qidx}, asym_sim={sim[qidx]}")
        # logger.info(f"qidx={qidx}, empirical_mean={empirical_mean[qidx]}")
    embeds_hinge_similarity_scores = hinge_sim(query_embeds, corpus_embeds)
    embeds_cosine_similarity_scores = cosine_similarity(query_embeds, corpus_embeds)
    logger.debug(f"Computed cosine similarity scores using EMBEDDINGS, shape={embeds_cosine_similarity_scores.shape}")
    fmaps_cosine_similarity_scores = cosine_similarity(query_fmaps, corpus_fmaps)
    logger.debug(f"Computed cosine similarity scores using FOURIER MAPS, shape={fmaps_cosine_similarity_scores.shape}")
    empirical_mean = (1 / 2 / np.pi / av.m_use * corpus_fmaps @ query_fmaps.T).T
    logger.debug(f"Computed empirical mean, shape={empirical_mean.shape}")
    sim = np.column_stack(sim).T
    logger.debug(f"Computed sim, shape={sim.shape}")
    all_d = {
        "asym_sim": sim,
        "empirical_mean": empirical_mean,
        "embeds_cosine_similarity": embeds_cosine_similarity_scores,
        "fmaps_cosine_similarity": fmaps_cosine_similarity_scores
    }
    # compare = np.vstack(compare)
    # logger.debug(f"compare.shape: {compare.shape}, compare.min: {np.min(compare)}, compare.max: {np.max(compare)}")
    plt.figure()
    plt.scatter(sim.reshape(-1), empirical_mean.reshape(-1))
    plt.title("Actual score vs Estimated Score (Using sampling)")
    plt.savefig("test_fourier.png")
    pickle.dump(all_d, open("../test_fourier_map.pkl", "wb"))
    fmaps_ktaus = []
    fmaps_aps = []
    embeds_cos_ktaus = []
    embeds_cos_aps = []
    embeds_hinge_ktaus = []
    embeds_hinge_aps = []
    for qidx in range(num_q):
        pred = ground_truth[qidx]
        pred = np.array([1 if c in pred else 0 for c in range(num_c)])
        fmaps_ktaus.append(get_ktau(sim[qidx], fmaps_cosine_similarity_scores[qidx]))
        fmaps_aps.append(get_ap(pred, fmaps_cosine_similarity_scores[qidx]))
        embeds_cos_ktaus.append(get_ktau(sim[qidx], embeds_cosine_similarity_scores[qidx]))
        embeds_cos_aps.append(get_ap(pred, embeds_cosine_similarity_scores[qidx]))
        embeds_hinge_ktaus.append(get_ktau(sim[qidx], embeds_hinge_similarity_scores[qidx]))
        embeds_hinge_aps.append(get_ap(pred, embeds_hinge_similarity_scores[qidx]))
        logger.info(f"qidx={qidx}: abs_pos_cnt={np.sum(pred)}, %_pos_cnt={100*np.sum(pred)/len(sim[qidx])}%, \
                fmaps_ktau={fmaps_ktaus[-1]}, fmaps_ap={fmaps_aps[-1]}, \
                embeds_cos_ktau={embeds_cos_ktaus[-1]}, embeds_cos_ap={embeds_cos_aps[-1]}, \
                embeds_hinge_ktau={embeds_hinge_ktaus[-1]}, embeds_hinge_ap={embeds_hinge_aps[-1]}")
    logger.info(f"fmaps avg. ktau = {np.mean(fmaps_ktaus)}, embeds cos avg. ktau = {np.mean(embeds_cos_ktaus)}, embeds hinge avg. ktau = {np.mean(embeds_hinge_ktaus)}")
    logger.info(f"fmaps MAP = {np.mean(fmaps_aps)}, embeds cos MAP = {np.mean(embeds_cos_aps)}, embeds hinge MAP = {np.mean(embeds_hinge_aps)}")

def test_fourier_map2(av):
    """
        1a. Fetch corupus embeddings
        1b. Fetch fourier map of corpus
        2a. Fetch query embeddings
        2b. Fetch fourier map of query
        3a. Compute asym_sim on (query, corpus) embeddings
        3b. Compute empirical mean using the samples
    """
    init_start = time.time()
    corpus_embeds_fetch_start = time.time()
    corpus_embeds = fetch_corpus_embeddings(av)
    corpus_embeds_fetch_time = time.time() - corpus_embeds_fetch_start
    logger.info(f"Corpus embeds shape: {corpus_embeds.shape}, time={corpus_embeds_fetch_time}")
    # assign_threshold(av, corpus_embeds)
    #This will init the k hash functions, each of dimension d
    lsh = LSH(av)
    # ws, pdfs = fetch_samples(av)
    if av.SPLIT == 'jointtrval':
        av.SPLIT = 'train'
        query_embeds_tr = fetch_query_embeddings(av)
        ground_truth_tr = fetch_ground_truths(av)
        av.SPLIT = 'val'
        query_embeds_val = fetch_query_embeddings(av)
        ground_truth_val = fetch_ground_truths(av)
        query_embeds = np.vstack((query_embeds_tr, query_embeds_val))
        ground_truth = {**{i:ground_truth_tr[i] for i in range(len(ground_truth_tr))}, **{i+len(ground_truth_tr):ground_truth_val[i] for i in range(len(ground_truth_val))}}
        av.SPLIT = 'jointtrval'
    else:
        query_embeds = fetch_query_embeddings(av)
        ground_truth = fetch_ground_truths(av)
    # ground_truth = fetch_ground_truths(av)
    logger.info('Ground truth fetched.')
    corpusfmaps_start_time = time.time()

    # synthT = [-1 for i in range(query_embeds.shape[1])]
    # for idx in range(query_embeds.shape[0]): 
    #     cur_max = np.max(np.abs(query_embeds[idx] - corpus_embeds), axis=1)
    #     synthT = [max(cur_max[i], synthT[i]) for i in range(query_embeds.shape[1])]

    # print(f"synthT: {synthT}")
    # synthT = np.max(synthT)
    # synthT = int(np.ceil(synthT))

    batch_sz = 10000
    # batches = []
    # for i in range(0, corpus_embeds.shape[0],batch_sz):
    #     batches.append(corpus_embeds[i:i+batch_sz])
    # assert sum([item.shape[0] for item in batches]) == corpus_embeds.shape[0]
    num_q = query_embeds.shape[0]
    num_c = corpus_embeds.shape[0]
    fmaps_init = time.time()
    if av.TASK == 'hinge':
        corpus_fmaps = np.zeros((corpus_embeds.shape[0], av.m_use * corpus_embeds.shape[1] * 4))
        logger.info(f"init corpus fmaps, shape={corpus_fmaps.shape}, time={time.time()-fmaps_init}")
        for i in tqdm.tqdm(range(0, corpus_embeds.shape[0],batch_sz)):
            corpus_fmaps[i:i+batch_sz,:] = lsh.generate_fmap(av, corpus_embeds[i:i+batch_sz], isQuery=False)
    
        # hashcodes = torch.cat(hcode_list)
        # corpus_fmaps = np.vstack(fmaps_list) #lsh.generate_fmap(av, corpus_embeds, isQuery=False)
        corpusfmaps_time = time.time() - corpusfmaps_start_time
        logger.info(f"Corpus fmaps shape: {corpus_fmaps.shape}, time={corpusfmaps_time}")
    # query_embeds = fetch_query_embeddings(av)
        queryfmaps_start_time = time.time()
        query_fmaps = lsh.generate_fmap(av, query_embeds, isQuery=True)
        queryfmaps_time = time.time() - queryfmaps_start_time
        logger.info(f"Query fmaps shape: {query_fmaps.shape}, time={queryfmaps_time}")
    elif av.TASK == 'dot':
        max_norm = np.max(np.linalg.norm(corpus_embeds, axis=1))
        d = corpus_embeds.shape[1]
        batch_sz  = 50000
        #Writing split manually to ensure correctness
        batches = []
        for i in range(0, corpus_embeds.shape[0],batch_sz):
            batches.append(corpus_embeds[i:i+batch_sz])
        assert sum([item.shape[0] for item in batches]) == corpus_embeds.shape[0]
        corpus_fmaps = np.zeros((corpus_embeds.shape[0], d+1))
        for i in range(0, corpus_embeds.shape[0],batch_sz):
            #NOTE: for query it will ultimately append 0, but we may directly do that under if isQuery to speed up
            batch_item_scaled = corpus_embeds[i:i+batch_sz]/max_norm
            app = np.expand_dims(np.sqrt(1-np.square(np.linalg.norm(batch_item_scaled, axis=1))),axis=-1)
            batch_item_augmented = np.hstack((batch_item_scaled,app))
            corpus_fmaps[i:i+batch_sz,:] = batch_item_augmented
        
        batch_item_scaled = query_embeds / np.linalg.norm(query_embeds, axis=1).reshape(num_q, 1)

        # logger.info(np.linalg.norm(batch_item_scaled, axis=1), np.sqrt(1-np.square(np.linalg.norm(batch_item_scaled, axis=1))))
        app = np.expand_dims(np.zeros(num_q),axis=-1)
        # logger.info(app)
        batch_item_augmented = np.hstack((batch_item_scaled,app))
        query_fmaps = batch_item_augmented
    else:
        corpus_fmaps = corpus_embeds
        query_fmaps = query_embeds
    init_time = time.time() - init_start

    # check if norms are constant
    norm_start = time.time()
    if av.TASK == 'hinge':
        v = np.linalg.norm(corpus_fmaps[0])
        # assert np.all(np.isclose(v, np.linalg.norm(corpus_fmaps, axis=1)))
        # norm1_time = time.time() - norm_start
        # logger.info(f"Successfully verified norm consistency of corpus fmaps, time={norm1_time}")
        norm2_start = time.time()
        assert np.all(np.isclose(v, np.linalg.norm(query_fmaps, axis=1)))
        norm2_time = time.time() - norm2_start
        logger.info(f"Successfully verified norm consistency of query fmaps, time={norm2_time}")
    norm_time = time.time() - norm_start
    # for i in range(num_c):
    #     assert np.isclose(np.linalg.norm(corpus_fmaps[i]), v), f"v={v}, corpus={np.linalg.norm(corpus_fmaps[i])}"
    # for i in range(num_q):
    #     assert np.isclose(np.linalg.norm(query_fmaps[i]), v), f"v={v}, query={np.linalg.norm(query_fmaps[i])}"

    # sim = [None for _ in range(num_q)]
    # empirical_mean = [None for _ in range(num_q)]
    # for qidx in range(num_q):
    #     logger.info("#"*18 + f" qidx={qidx} " + "#"*18)
    #     sim[qidx] = asym_sim(av, query_embeds[qidx], corpus_embeds)
        # logger.info(f"sim.shape = {sim[qidx].shape}")
        # compare[qidx] = np.column_stack((sim[qidx], empirical_mean[qidx]))
        # logger.info(f"qidx={qidx}, compare={compare[qidx]}")
        # logger.info(f"qidx={qidx}, asym_sim={sim[qidx]}")
        # logger.info(f"qidx={qidx}, empirical_mean={empirical_mean[qidx]}")
    embeds_hinge_start = time.time()
    scoring_fns = {"hinge": hinge_sim, "cos": cosine_similarity, "dot": dot_sim}
    embeds_hinge_similarity_scores = scoring_fns[av.TASK](query_embeds, corpus_embeds)
    embeds_hinge_time = time.time() - embeds_hinge_start
    logger.info(f"Computed embeds {av.TASK} sim score, shape={embeds_hinge_similarity_scores.shape}, time={embeds_hinge_time}")
    # embeds_cosine_similarity_scores = cosine_similarity(query_embeds, corpus_embeds)
    # logger.debug(f"Computed cosine similarity scores using EMBEDDINGS, shape={embeds_cosine_similarity_scores.shape}")
    fmaps_cos_start = time.time()
    scoring_fns2 = {"hinge": normalized_cosine_similarity, "cos": cosine_similarity, "dot": dot_sim}
    fmaps_cosine_similarity_scores = scoring_fns2[av.TASK](query_fmaps, corpus_fmaps)
    hplanes = {'hinge': lsh.gauss_hplanes_fhash, 'cos': lsh.gauss_hplanes_cos, 'dot': lsh.gauss_hplanes_dot}
    projected_query_fmaps = query_fmaps @ hplanes[av.TASK]
    logger.info(f"Query projected fmaps shape: {projected_query_fmaps.shape}")
    projected_corpus_fmaps = corpus_fmaps @ hplanes[av.TASK]
    logger.info(f"Corpus projected fmaps shape: {projected_corpus_fmaps.shape}")
    projected_hinge_similarity_scores = scoring_fns2[av.TASK](projected_query_fmaps, projected_corpus_fmaps)
    sign_hinge_similarity_scores = scoring_fns2[av.TASK](np.sign(projected_query_fmaps), np.sign(projected_corpus_fmaps))
    fmaps_cos_time = time.time() - fmaps_cos_start
    logger.info(f"Computed cosine similarity scores using FOURIER MAPS, dot pdt using projections and signed projections, shape={fmaps_cosine_similarity_scores.shape}, {projected_hinge_similarity_scores.shape}, {sign_hinge_similarity_scores.shape}, time={fmaps_cos_time}")
    # empirical_mean = (1 / 2 / np.pi / av.m_use * corpus_fmaps @ query_fmaps.T).T
    # logger.debug(f"Computed empirical mean, shape={empirical_mean.shape}")
    # sim = np.column_stack(sim).T
    # logger.debug(f"Computed sim, shape={sim.shape}")
    # all_d = {
    #     "asym_sim": sim,
    #     "empirical_mean": empirical_mean,
    #     "embeds_cosine_similarity": embeds_cosine_similarity_scores,
    #     "fmaps_cosine_similarity": fmaps_cosine_similarity_scores
    # }
    # compare = np.vstack(compare)
    # logger.debug(f"compare.shape: {compare.shape}, compare.min: {np.min(compare)}, compare.max: {np.max(compare)}")
    # plt.figure()
    # plt.scatter(sim.reshape(-1), empirical_mean.reshape(-1))
    # plt.title("Actual score vs Estimated Score (Using sampling)")
    # plt.savefig("test_fourier.png")
    # pickle.dump(all_d, open("../test_fourier_map.pkl", "wb"))
    eval_start = time.time()
    fmaps_ktaus = []
    fmaps_aps = []
    projected_fmaps_ktaus = []
    projected_fmaps_aps = []
    sign_fmaps_ktaus = []
    sign_fmaps_aps = []
    # embeds_cos_ktaus = []
    # embeds_cos_aps = []
    # embeds_hinge_ktaus = []
    # embeds_hinge_aps = []
    for qidx in range(num_q):
        pred = ground_truth[qidx]
        pred = np.array([1 if c in pred else 0 for c in range(num_c)])
        fmaps_ktaus.append(get_ktau(embeds_hinge_similarity_scores[qidx], fmaps_cosine_similarity_scores[qidx]))
        fmaps_aps.append(get_ap(pred, fmaps_cosine_similarity_scores[qidx]))
        projected_fmaps_ktaus.append(get_ktau(embeds_hinge_similarity_scores[qidx], projected_hinge_similarity_scores[qidx]))
        projected_fmaps_aps.append(get_ap(pred, projected_hinge_similarity_scores[qidx]))
        sign_fmaps_ktaus.append(get_ktau(embeds_hinge_similarity_scores[qidx], sign_hinge_similarity_scores[qidx]))
        sign_fmaps_aps.append(get_ap(pred, sign_hinge_similarity_scores[qidx]))
        # embeds_cos_ktaus.append(get_ktau(sim[qidx], embeds_cosine_similarity_scores[qidx]))
        # embeds_cos_aps.append(get_ap(pred, embeds_cosine_similarity_scores[qidx]))
        # embeds_hinge_ktaus.append(get_ktau(sim[qidx], embeds_hinge_similarity_scores[qidx]))
        # embeds_hinge_aps.append(get_ap(pred, embeds_hinge_similarity_scores[qidx]))
        # logger.info(f"qidx={qidx}: abs_pos_cnt={np.sum(pred)}, %_pos_cnt={100*np.sum(pred)/len(sim[qidx])}%, \
        #         fmaps_ktau={fmaps_ktaus[-1]}, fmaps_ap={fmaps_aps[-1]}, \
        #         embeds_cos_ktau={embeds_cos_ktaus[-1]}, embeds_cos_ap={embeds_cos_aps[-1]}, \
        #         embeds_hinge_ktau={embeds_hinge_ktaus[-1]}, embeds_hinge_ap={embeds_hinge_aps[-1]}")
    eval_time = time.time() - eval_start
    logger.info(f"init={init_time}, norm_check={norm_time}, embeds_hinge={embeds_hinge_time}, fmaps_cos={fmaps_cos_time}, eval={eval_time}")
    logger.info(f"fmaps avg. ktau = {np.mean(fmaps_ktaus)}, projected avg. ktau={np.mean(projected_fmaps_ktaus)}, sign avg. ktau={np.mean(sign_fmaps_ktaus)}")
    logger.info(f"fmaps MAP = {np.mean(fmaps_aps)}, projected avg. map={np.mean(projected_fmaps_aps)}, sign avg. map={np.mean(sign_fmaps_aps)}")
    return fmaps_ktaus, fmaps_aps

def get_labels(actual_score, av):
    pred = np.zeros(len(actual_score))
    threshold = (av.T - av.T1) * av.embed_dim
    pred[(actual_score >= threshold)] = 1
    return pred

def get_ktau(actual_score, estimated_score):
    return stats.kendalltau(estimated_score, actual_score)[0]

def get_ap(pred, estimated_score):
    map_ = average_precision_score(pred, estimated_score)
    return map_


def assign_threshold(av, corpus_embeds):
    """
        TODO: heuristic here to set T for probability sampling 
    """
    av.T = 2 * np.max(np.abs(corpus_embeds/av.SCALE))

def set_seed():
    # Set random seeds
  seed = 4
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)
  torch.backends.cudnn.deterministic = False



def elim_excess_cids(av,corpus_embeds, ground_truth, num_cid_remove):
    av_split = av.SPLIT
    av.SPLIT = "train"
    tr_gt = fetch_ground_truths(av)

    #TODO: filter out corpus embeds
    num_c = corpus_embeds.shape[0]
    all_pos = []
    for qid in tr_gt.keys():
        all_pos.extend(tr_gt[qid])
    all_neg_cids = list(set(list(range(num_c))) - set(all_pos))
    neg_cids_5k = all_neg_cids[:num_cid_remove]
    new_corpus_embeds = np.delete(corpus_embeds,neg_cids_5k,axis=0)

    cid_map_dict = {}
    new_cidx = 0
    for old_qidx in range(num_c): 
        if old_qidx not in set(neg_cids_5k):
            cid_map_dict[old_qidx] = new_cidx
            new_cidx+=1

    new_ground_truth = {}
    for qid in ground_truth.keys():
        new_ground_truth[qid] = []
        for cid in ground_truth[qid]:
            if cid in cid_map_dict: 
                new_ground_truth[qid].append(cid_map_dict[cid]) 

    av.SPLIT = av_split
    #ground_truth = new_ground_truth
    return new_corpus_embeds, new_ground_truth 
    #############################################################33

def run_lsh(av,num_cid_remove=0):
    """
        1. fetch hashcode and embeds of corpus items
        2. send to lsh class for bucket creation 
        3. Loop over query items
        4. Each time send query embed, hashcode, k to LSH
        5. LSH returns top k items and time to compute the same
        6. compute precision@k and note the time taken. 
        7. This performance eval and time will be compared to brute force and random distribution alternatives. 
    """
    corpus_embeds = fetch_corpus_embeddings(av)
    #TODO: insert heuristic here to set vals of T and T1 . Ideally q embeds should not be seen here
    # assign_threshold(av,corpus_embeds)
    #Fetch queries and ground truth for performance eval
    if av.SPLIT == 'jointtrval':
        av.SPLIT = 'train'
        query_embeds_tr = fetch_query_embeddings(av)
        ground_truth_tr = fetch_ground_truths(av)
        av.SPLIT = 'val'
        query_embeds_val = fetch_query_embeddings(av)
        ground_truth_val = fetch_ground_truths(av)
        query_embeds = np.vstack((query_embeds_tr, query_embeds_val))
        ground_truth = {**{i:ground_truth_tr[i] for i in range(len(ground_truth_tr))}, **{i+len(ground_truth_tr):ground_truth_val[i] for i in range(len(ground_truth_val))}}
        av.SPLIT = 'jointtrval'
    else:
        query_embeds = fetch_query_embeddings(av)
        ground_truth = fetch_ground_truths(av)

    new_corpus_embeds, new_ground_truth = elim_excess_cids(av,corpus_embeds, ground_truth, num_cid_remove)
  
    num_qitems = query_embeds.shape[0]
    assert(len(ground_truth.keys()) == num_qitems)

    set_seed()
    #This will init the k hash functions, each of dimension d
    lsh = LSH(av)
    #This will generate feature maps and index corpus items
    lsh.index_corpus(new_corpus_embeds,av.HASH_MODE)


    all_hashing_info_dict = {}
    all_hashing_info_dict['asymhash']= []
    all_hashing_info_dict['nohash']= []
    
    for qid,qemb in enumerate(query_embeds): 
        #reshape qemb to 1*d
        qemb_reshaped = np.expand_dims(qemb, axis=0)
        all_hashing_info_dict['asymhash'].append(lsh.retrieve(qemb_reshaped,av.K, hash_mode=av.HASH_MODE, no_bucket=False,qid=qid))
    
    for qid,qemb in enumerate(query_embeds): 
        #reshape qemb to 1*d
        qemb_reshaped = np.expand_dims(qemb, axis=0)
        all_hashing_info_dict['nohash'].append(lsh.retrieve(qemb_reshaped,av.K,hash_mode=None, no_bucket=True,qid=qid))
  
        

    # =====================HASHING PERF Precision@K ================================
    all_patk_hash = []
    all_patk_nohash = []
    all_customap_hash = []
    all_customap_nohash = []
    
    for qidx in range(len(query_embeds)):
        all_patk_hash.append(precision_at_k(set(new_ground_truth[qidx]),set(all_hashing_info_dict['asymhash'][qidx][2]),av.K))
        all_customap_hash.append(custom_ap(set(new_ground_truth[qidx]),all_hashing_info_dict['asymhash'][qidx][2],all_hashing_info_dict['asymhash'][qidx][1],av.K, len(set(ground_truth[qidx]))))
        all_patk_nohash.append(precision_at_k(set(new_ground_truth[qidx]),set(all_hashing_info_dict['nohash'][qidx][2]),av.K))
        all_customap_nohash.append(custom_ap(set(new_ground_truth[qidx]),all_hashing_info_dict['nohash'][qidx][2],all_hashing_info_dict['nohash'][qidx][1],av.K, len(set(ground_truth[qidx]))))
    
    lsh.pretty_print_hash_tables(20)

    logger.info("hashing P@K averaged across all queries is %s", np.mean(all_patk_hash) )
    logger.info("Exhaustive search P@K averaged across all queries is %s", np.mean(all_patk_nohash) )
    logger.info("hashing Custom AP averaged across all queries is %s", np.mean(all_customap_hash) )
    logger.info("Exhaustive search Custom AP averaged across all queries is %s", np.mean(all_customap_nohash) )

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
    
    logger.info("Time stats for exhaustive search")
    for k,v in time_logger_dict['nohash'].items():
        logger.info (f"{k}, {v}")
    for k,v in other_data_dict['nohash'].items():
        logger.info (f"{k}, {v}")
    
    logger.info("Time stats for hashing")
    for k,v in time_logger_dict['asymhash'].items():
        logger.info (f"{k}, {v}")
    for k,v in other_data_dict['asymhash'].items():
        logger.info (f"{k}, {v}")
    
    for k in time_logger_dict['nohash'].keys():
        logger.info(f"Total {k} Nohash: {sum(list(time_logger_dict['nohash'][k].values())) - time_logger_dict['nohash'][k]['take_time']}, Hash: {sum(list(time_logger_dict['asymhash'][k].values()))-time_logger_dict['asymhash'][k]['take_time']}")
    # return np.mean(all_patk_hash), np.mean(all_patk_nohash),np.mean(all_customap_hash), np.mean(all_customap_nohash), time_logger_dict 
    return np.mean(all_customap_hash), np.mean(all_customap_nohash), time_logger_dict, all_customap_nohash, lsh, all_hashing_info_dict 
#def get_fourier_map_fname(av, type_):
#    """
#        type_ "Query"/"Corpus"
#    """
#    if av.TASK == "synth": 
#        fp = "./data/FourierMap" + type_ + "_task" + str(av.TASK) + "_delta" + str(av.delta) + "_embed_dim"+ str(av.embed_dim) +\
#                "_num_q" + str(av.num_q) + "_num_cpos_perq" + str(av.num_cpos_perq) + "_num_cneg_perq" +\
#                str(av.num_cneg_perq) + "_T" + str(av.T) + "_T1" + str(av.T1) +".pkl"
#        logger.info("Getting fourier map from %s", fp)
#        return fp
#    else: 
#        raise NotImplementedError()

def fetch_ground_truths(av):
    #logger.info('Fetching ground truth labels.')
    if av.DATASET_NAME == "syn": 
        fp = "./data/SyntheticData_delta" + str(av.delta) + "_embed_dim"+ str(av.embed_dim) +\
                        "_num_q" + str(av.num_q) + "_num_cpos_perq" + str(av.num_cpos_perq) + "_num_cneg_perq" +\
                        str(av.num_cneg_perq) + "_T" + str(av.synthT) + "_T1" + str(av.T1) +".pkl"
        if not os.path.exists(fp):
           generate_labels(av) 

        all_d = pickle.load(open(fp,"rb"))
        logger.info("Loading ground truth labels from %s", fp)
        return all_d['positive_labels']
    elif av.DATASET_NAME == "msweb":
        fp = f"./data/msweb{'_csubq' if av.CsubQ else ''}_embeds_{av.TASK}.pkl"
        all_d = pickle.load(open(fp,"rb"))
        logger.info("Loading ground truth labels from %s", fp)
        return all_d[f'{av.SPLIT}_positive_labels']
    elif av.DATASET_NAME in ["msnbc", "msweb2", "msweb3","msweb294","msnbc294","msnbc294_1","msnbc294_2","msnbc294_3","msnbc294_4","msnbc294_5","msnbc294_6","msnbc294_7","msweb294_1"]:
        fp = f"./data/{av.DATASET_NAME}{'_csubq' if av.CsubQ else ''}_embeds_{av.TASK}.pkl"
        all_d = pickle.load(open(fp,"rb"))
        logger.info("Loading ground truth labels from %s", fp)
        return all_d[f'{av.SPLIT}_positive_labels']
    elif av.DATASET_NAME in ["aids","ptc_mm","ptc_fm"]:
        fp = f"./data/{av.DATASET_NAME}_embeds_{av.TASK}.pkl"
        all_d = pickle.load(open(fp,"rb"))
        logger.info("Loading ground truth labels from %s", fp)
        return all_d[f'{av.SPLIT}_positive_labels']
    else: 
        raise NotImplementedError()

def store_hashing_result_info(av,data):
    """
        #TODO:IR
        This should include av.m_use, av.hcode_dim, av.subset_size, av.num_hash_tables, av.T, av.T1, av.synthT
    """
    if av.DATASET_NAME == "syn": 
        fp = "./data/HashingInfo_SyntheticData_delta" + str(av.delta) + "_embed_dim"+ str(av.embed_dim) +\
                        "_num_q" + str(av.num_q) + "_num_cpos_perq" + str(av.num_cpos_perq) + "_num_cneg_perq" +\
                        str(av.num_cneg_perq) + "_T" + str(av.synthT) + "_T1" + str(av.T1) +".pkl"
    elif av.DATASET_NAME == "msweb":
        fp = f"./data/HashingInfo_msweb{'_csubq' if av.CsubQ else ''}_{av.TASK}.pkl"
    elif av.DATASET_NAME in ["msnbc", "msweb2", "msweb3","msweb294","msnbc294","msnbc294_1","msnbc294_2","msnbc294_3","msnbc294_4","msnbc294_5","msnbc294_6","msnbc294_7","msweb294_1"]:
        fp = f"./data/HashingInfo_{av.DATASET_NAME}{'_csubq' if av.CsubQ else ''}_{av.TASK}.pkl"
    elif av.DATASET_NAME  in ["aids","ptc_mm","ptc_fm"]:
        fp = f"./data/HashingInfo_{av.DATASET_NAME}_{av.TASK}.pkl"
    else: 
        raise NotImplementedError()

    pickle.dump(data,open(fp,"wb"))

def fetch_query_embeddings(av):
    logger.info('Fetching query embeddings.')
    if av.DATASET_NAME == "syn": 
        #delta  = 15
        #embed_dim = 10
        #num_q = 200
        #num_c_perq = 1000
        #T=125
        #T1 = 10
        embed_fp = "./data/SyntheticData_delta" + str(av.delta) + "_embed_dim"+ str(av.embed_dim) +\
                        "_num_q" + str(av.num_q) + "_num_cpos_perq" + str(av.num_cpos_perq) + "_num_cneg_perq" +\
                        str(av.num_cneg_perq) + "_T" + str(av.synthT) + ".pkl" #+ "_T1" + str(av.T1) +".pkl"
        all_d = pickle.load(open(embed_fp,"rb"))
        logger.info("From %s", embed_fp)
        return all_d['all_q'].astype(dtype=np.float32)
    elif av.DATASET_NAME == "msweb":
        embed_fp = f"./data/msweb{'_csubq' if av.CsubQ else ''}_embeds_{av.TASK}.pkl"
        all_d = pickle.load(open(embed_fp,"rb"))
        logger.info("From %s", embed_fp)
        return all_d[f'{av.SPLIT}_q']
    elif av.DATASET_NAME in ["msnbc", "msweb2", "msweb3","msweb294","msnbc294","msnbc294_1","msnbc294_2","msnbc294_3","msnbc294_4","msnbc294_5","msnbc294_6","msnbc294_7","msweb294_1"]:
        embed_fp = f"./data/{av.DATASET_NAME}{'_csubq' if av.CsubQ else ''}_embeds_{av.TASK}.pkl"
        all_d = pickle.load(open(embed_fp,"rb"))
        logger.info("From %s", embed_fp)
        return all_d[f'{av.SPLIT}_q']
    elif av.DATASET_NAME in ["aids","ptc_mm","ptc_fm"]:
        embed_fp = f"./data/{av.DATASET_NAME}_embeds_{av.TASK}.pkl"
        all_d = pickle.load(open(embed_fp,"rb"))
        logger.info("From %s", embed_fp)
        return all_d[f'{av.SPLIT}_q']
    else: 
        raise NotImplementedError()

def fetch_corpus_embeddings(av):
    logger.info('Fetching corpus embeddings.')
    if av.DATASET_NAME == "syn": 
        #delta  = 15
        #embed_dim = 10
        #num_q = 200
        #num_c_perq = 1000
        #T=125
        #T1 = 10
        embed_fp = "./data/SyntheticData_delta" + str(av.delta) + "_embed_dim"+ str(av.embed_dim) +\
                        "_num_q" + str(av.num_q) + "_num_cpos_perq" + str(av.num_cpos_perq) + "_num_cneg_perq" +\
                        str(av.num_cneg_perq) + "_T" + str(av.synthT) + ".pkl" #+ "_T1" + str(av.T1) +".pkl"
    elif av.DATASET_NAME == "msweb":
        embed_fp = f"./data/msweb{'_csubq' if av.CsubQ else ''}_embeds_{av.TASK}.pkl"
    elif av.DATASET_NAME in ["msnbc", "msweb2", "msweb3", "msweb294","msnbc294","msnbc294_1","msnbc294_2","msnbc294_3","msnbc294_4","msnbc294_5","msnbc294_6","msnbc294_7","msweb294_1"]:
        embed_fp = f"./data/{av.DATASET_NAME}{'_csubq' if av.CsubQ else ''}_embeds_{av.TASK}.pkl"
    elif av.DATASET_NAME  in ["aids","ptc_mm","ptc_fm"]:
        embed_fp = f"./data/{av.DATASET_NAME}_embeds_{av.TASK}.pkl"
    else:
        raise NotImplementedError()
    
    all_d = pickle.load(open(embed_fp,"rb"))
    logger.info("From %s", embed_fp)
    return all_d['all_c'].astype(dtype=np.float32)


def run_falconn_lsh(av,query_embeds,corpus_embeds,ground_truth):

    #corpus_embeds = fetch_corpus_embeddings(av)
    ##Fetch queries and ground truth for performance eval
    #query_embeds = fetch_query_embeddings(av)
    #ground_truth = fetch_ground_truths(av)

    #num_qitems = query_embeds.shape[0]
    #assert(len(ground_truth.keys()) == num_qitems)

    
    set_seed()
    if av.HASH_MODE == "fhash":
        lsh = LSH(av)
        batch_sz = 40000
        corpus_fmaps = cudavar(av,torch.zeros((corpus_embeds.shape[0], av.m_use * corpus_embeds.shape[1] * 4)))
        logger.info(f"init corpus fmaps, shape={corpus_fmaps.shape}")
        for i in tqdm.tqdm(range(0, corpus_embeds.shape[0],batch_sz)):
            isQuery = True if av.SPECIAL_FHASH == "SymQuery" else False
            corpus_fmaps[i:i+batch_sz,:] = torch.from_numpy(lsh.generate_fmap(av, corpus_embeds[i:i+batch_sz], isQuery)).type(torch.float)
        
        isQuery = False if av.SPECIAL_FHASH == "SymCorpus" else True
        query_fmaps = cudavar(av,torch.from_numpy(lsh.generate_fmap(av, query_embeds, isQuery)).type(torch.float))
        
        if not av.FMAP_SELECT == -1:
            corpus_data = corpus_fmaps.numpy().astype(np.float32)[:,:av.FMAP_SELECT]
            query_data = query_fmaps.numpy().astype(np.float32)[:,:av.FMAP_SELECT]
        else:
            corpus_data = corpus_fmaps.numpy().astype(np.float32)
            query_data = query_fmaps.numpy().astype(np.float32)            
       
    elif av.HASH_MODE == "cosine":
        corpus_data = corpus_embeds.astype(np.float32)
        query_data = query_embeds.astype(np.float32)
    
    #TODO: center
    #corpus_data = corpus_data -np.mean(corpus_data, axis=0)[None,:]
    #print(np.mean(corpus_data, axis=0).shape)
    #print(np.linalg.norm(corpus_data, axis=1, keepdims=True) )
    #print(corpus_data.shape)
    #corpus_data  = corpus_data/(np.linalg.norm(corpus_data, axis=1, keepdims=True))  
    #TODO: center
    #query_data = query_data - np.mean(query_data, axis=0)[None,:] 
    #query_data  = query_data/np.linalg.norm(query_data, axis=1, keepdims=True)  

    s = time.time()
    p = falconn.get_default_parameters(corpus_data.shape[0], corpus_data.shape[1])
    p.dimension = corpus_data.shape[1]
    p.k  = av.subset_size
    p.l = av.num_hash_tables
    p.lsh_family = falconn.LSHFamily.Hyperplane
    p.seed = 4
    p.storage_hash_table =  falconn.StorageHashTable.LinearProbingHashTable
    p.distance_function = falconn.DistanceFunction.EuclideanSquared
    t = falconn.LSHIndex(p)
    t.setup(corpus_data)
    print(f"corpus indexing time: {time.time()-s}")
    
    q = t.construct_query_object()
    q.set_num_probes(av.num_hash_tables)

 

    all_hashing_info_dict = {}
    all_hashing_info_dict['asymhash']= []
    
    s = time.time()
    for qemb in query_data: 
        #reshape qemb to 1*d
        qemb_reshaped = np.expand_dims(qemb, axis=0)
        all_hashing_info_dict['asymhash'].append(q.get_unique_candidates(qemb))
     
    print(f"query time: {time.time()-s}")


    # =====================HASHING PERF Precision@K ================================
    all_patk_hash = []
    all_customap_hash = []

    s = time.time()

    for qidx in range(len(query_embeds)):
        all_patk_hash.append(precision_at_k(set(ground_truth[qidx]),set(all_hashing_info_dict['asymhash'][qidx]),av.K))
        #cdata = np.take(corpus_data,list(all_hashing_info_dict['asymhash'][qidx]),axis=0)
        cdata = np.take(corpus_embeds,list(all_hashing_info_dict['asymhash'][qidx]),axis=0)

        #print(cdata.shape)
        #qdata = query_data[qidx][None,:]
        qdata = query_embeds[qidx][None,:]
        #print(qdata.shape)
        #scores = (qdata*cdata).sum(-1)
        if av.TASK == "hinge":
            scores = asym_sim(av,qdata,cdata)
        elif av.TASK == "cos":
            scores = optimized_cosine_similarity(qdata,cdata).reshape(-1)

        #print(scores.shape)
        #print(len(all_hashing_info_dict['asymhash'][qidx]))
        all_customap_hash.append(custom_ap(set(ground_truth[qidx]),all_hashing_info_dict['asymhash'][qidx],list(scores),av.K))
    print(f"scoring time: {time.time()-s}")

    logger.info("hashing P@K averaged across all queries is %s", np.mean(all_patk_hash) )
    logger.info("hashing Custom AP averaged across all queries is %s", np.mean(all_customap_hash) )

    return np.mean(all_patk_hash), np.mean(all_customap_hash) 


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                 type=str,   default="logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",               type=bool,  default=False)
  ap.add_argument("--has_cuda",                type=bool,  default=torch.cuda.is_available())
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
  ap.add_argument("--m_load",                  type=int,   default=10000)
  ap.add_argument("--m_use",                   type=int,   default=1000)
  ap.add_argument("--a",                       type=int,   default=-100)
  ap.add_argument("--b",                       type=int,   default=100)
  ap.add_argument("--hcode_dim",               type=int,   default=16)
  ap.add_argument("--num_hash_tables",         type=int,   default=10)
  ap.add_argument("--subset_size",             type=int,   default=8)
  ap.add_argument("--synthT",                  type=int,   default=38) 
  #ap.add_argument("--synthT",                  type=int,   default=133) 
  ap.add_argument("--SCALE",                   type=int,   default=1) 
  ap.add_argument("--sigmoid_a",               type=float,   default=-3.3263) 
  ap.add_argument("--sigmoid_b",               type=float,   default=3.7716) 
  ap.add_argument("--T",                       type=float,   default=37) 
  ap.add_argument("--T1",                      type=float,   default=0.1)
  #ap.add_argument("--T1",                      type=int,   default=1)
  ap.add_argument("--NUM_CID_REMOVE",          type=int,   default=0)
  ap.add_argument("--K",                       type=int,   default=-1)
  ap.add_argument("--DIR_PATH",                type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--pickle_fp",               type=str,   default="",help="path/to/datasets")
  ap.add_argument("--trained_cosine_fmap_pickle_fp",  type=str,   default="",help="path/to/datasets")
  #ap.add_argument("--P2N",                     type=float, default=1.0)    
  ap.add_argument("--DATASET_NAME",            type=str,   default="syn", help="syn/msnbc/msweb/graphs")
  ap.add_argument("--TASK",                    type=str,   default="hinge", help="cos/hinge/dot/dotperquery")
  ap.add_argument("--CsubQ",                          action='store_true')
  ap.add_argument("--HASH_MODE",               type=str,   default="fhash", help="fhash/cosine/dot")
  ap.add_argument("--SPLIT",                   type=str,   default="test", help="train/val/test/jointtrval/all")
  ap.add_argument("--SPECIAL_FHASH",           type=str,   default="", help="")
  ap.add_argument("--FMAP_SELECT",           type=int,   default=-1, help="")
  ap.add_argument("--E2EVER",            type=str,   default="", help="")
  ap.add_argument("--DEBUG",                   action='store_true')
  ap.add_argument("--use_pretrained_fmap",      action='store_true')
  ap.add_argument("--use_pretrained_hcode",    action='store_true')
  ap.add_argument("--HIDDEN_LAYERS",               type=int, nargs='*',  default=[])
  ap.add_argument("--FMAP_HIDDEN_LAYERS",               type=int, nargs='*',  default=[])
  ap.add_argument("--NO_TANH",                  action='store_true')
  ap.add_argument("--SIGN_EVAL",                  action='store_true')
  ap.add_argument("--INIT_GAUSS",                  action='store_true')
  ap.add_argument("--LAYER_NORMALIZE",                action='store_true')
  ap.add_argument("--USE_FMAP_BCE",             action='store_true')
  ap.add_argument("--USE_FMAP_BCE2",             action='store_true')
  ap.add_argument("--USE_FMAP_BCE3",             action='store_true')
  ap.add_argument("--USE_FMAP_PQR",             action='store_true')
  ap.add_argument("--DESC",           type=str,   default="", help="")
  ap.add_argument("--MARGIN",                  type=float, default=1.0)    
  ap.add_argument("--SCLOSS_MARGIN",           type=float, default=1.0)    
  ap.add_argument("--LOSS_TYPE",               type=str,   default="cos_ap", help="cos_ap/dot_ce")
  ap.add_argument("--FMAP_LOSS_TYPE",               type=str,   default="cos_ap", help="cos_ap/dot_ce")
  ap.add_argument("--TANH_TEMP",               type=float, default=1.0)    
  ap.add_argument("--FENCE_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--DECORR_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--C1_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--WEAKSUP_LAMBDA",            type=float, default=0.0)    
  ap.add_argument("--ES",                      type=int,   default=5)      
  ap.add_argument("--tr_fmap_dim",               type=int,   default=10)

  av = ap.parse_args()

  av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+"_hashing"
  if av.DEBUG: 
    av.logpath = av.logpath + "_DEBUG"

  if av.DATASET_NAME == "syn":
    #av.T is determined by code. So not set here  
    av.logpath = av.logpath + "_SynthT_" + str(av.synthT) + "_T1_" + str(av.T1)
  
  av.logpath = av.logpath + "_Muse_" + str(av.m_use) + "_MLoad_" + str(av.m_load) +\
                "_HCodeDim_" + str(av.hcode_dim) + "_NumHTables_" + str(av.num_hash_tables) +\
                "_SubsetSize_" + str(av.subset_size) + "_K_" + str(av.K) 

  set_log(av)
  logger.info("Command line")  
  logger.info('\n'.join(sys.argv[:]))  
  
  # Set random seeds
  seed = 4
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)
  torch.backends.cudnn.deterministic = False

  if av.DEBUG:
    av.m_load = av.m_use * av.embed_dim
    test_fourier_map2(av)
    # test_fmap2_data = {}
    # for num_samples in list(range(1000,11000,1000)):
    #     test_fmap2_data[num_samples] =  {}
    #     for T in  [1,2,3,4,5,7,9,11,13,15]:
    #         av.T = T
    #         av.m_use = num_samples
    #         av.m_load = av.embed_dim*av.m_use
    #         test_fmap2_data[num_samples][T] = test_fourier_map2(av)
    #         pickle.dump(test_fmap2_data,open(f"{av.DATASET_NAME}_{av.SPLIT}_test_fmap2_data.pkl","wb"))
  else:
    # tdicts = []
    # for i in range(5):
    #     # Set random seeds
    #     seed = 4
    #     random.seed(seed)
    #     np.random.seed(seed + 1)
    #     torch.manual_seed(seed + 2)
    #     torch.backends.cudnn.deterministic = False
    #     _, _, tdict = run_lsh(av)
    #     tdicts.append(tdict)
    # for tdict in tdicts:
    #     print(tdict)
    # av.T = 3
    av.m_load = av.m_use * av.embed_dim
    run_lsh(av,av.NUM_CID_REMOVE)
    # lsh_m_search = {}
    # for num_samples in list(range(1000,11000,1000)):
    #     av.T = 3
    #     av.m_use = num_samples
    #     av.m_load = av.embed_dim*av.m_use
    #     hash_map, nohash_map, tdict = run_lsh(av)
    #     lsh_m_search[num_samples] = {'hash_map': hash_map, 'nohash_map': nohash_map, 'tdict': tdict}


