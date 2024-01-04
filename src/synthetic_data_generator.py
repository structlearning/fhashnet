import numpy as np 
from numpy import linalg as LA
import time
import matplotlib.pyplot as plt
import itertools
import argparse
import pickle
import os
from common import logger, set_log

def asym(a,b):
    return np.sum(np.maximum(0,a-b))

def sym(a,b):
    return np.sum(np.abs(a-b))

def generate_queries(av):
    #If we do this, then some queries are close to one another under asym distance measure
    #hq = np.random.uniform(low=-delta,high=delta,size=(num_q,embed_dim))

    #Instead we do this
    #Set some high threshold for pairwise distance between sampled queries
    distinct_q = []
    #This hould be at least greater than the similarity threshold T1 (used later)
    q_thresh = av.q_thresh
    s = time.time()
    while (len(distinct_q)<av.num_q): 
        proposed_q = np.random.uniform(low=-av.delta,high=av.delta,size=(av.embed_dim))
        dup = False 
        for existing_q in distinct_q:
            if (asym(existing_q,proposed_q)<q_thresh) or (asym(proposed_q,existing_q)<q_thresh):
                dup = True
        if not dup:
            distinct_q.append(proposed_q)
    print(f"Time to generate queries: {time.time()-s}")

    all_q = np.vstack(distinct_q)
    return all_q

def generate_corpus(av, all_q):
    #sampling corpus graphs 
    list_hc = []
    asym_q_pos = []
    asym_q_neg = []
    sym_q_pos = []
    sym_q_neg = []

    for qidx in range(av.num_q):
        # hc=  np.random.multivariate_normal(all_q[qidx], 0.05*delta*np.eye(embed_dim), num_c_perq)
        hcpos =  all_q[qidx] + np.abs(np.random.uniform(av.pos_low, high=av.pos_high, size=(av.num_cpos_perq, av.embed_dim)))
        for hc in hcpos:
            asym_q_pos.append(asym(all_q[qidx],hc))
            sym_q_pos.append(sym(all_q[qidx],hc))
        hcneg =  all_q[qidx] - np.abs(np.random.uniform(av.neg_low, high=av.neg_high, size=(av.num_cneg_perq, av.embed_dim)))
        for hc in hcneg:
            asym_q_neg.append(asym(all_q[qidx],hc))
            sym_q_neg.append(sym(all_q[qidx],hc))
        list_hc.append(hcpos)
        list_hc.append(hcneg)

    all_c = np.vstack(list_hc)
    return all_c

def get_synthT(all_q, all_c):
    synthT = [-1 for i in range(all_q.shape[1])]
    for idx in range(all_q.shape[0]): 
        cur_max = np.max(np.abs(all_q[idx] - all_c), axis=1)
        synthT = [max(cur_max[i], synthT[i]) for i in range(all_q.shape[1])]

    print(f"synthT: {synthT}")
    synthT = np.max(synthT)
    synthT = int(np.ceil(synthT))
    return synthT

#def get_T1():
#    #This T1 should be less than the pairwise query distance used earlier (20)
#    T1 = 10
#    return T1

def assign_labels(all_q, all_c, T, T1):
    labels = {i: [] for i in range(all_q.shape[0])}
    corpus_ids = np.array([i for i in range(all_c.shape[0])])
    #Per query positive label count
    for idx in range(all_q.shape[0]): 
        sim = np.sum(T - np.maximum(0, all_q[idx] - all_c), axis=1)
        threshold = (T - T1) * all_q.shape[1]
        labels[idx] = corpus_ids[sim >= threshold]
        abs_cnt = sum(sim >= threshold)

        sym_dist = np.sum(np.abs(all_q[idx] - all_c), axis=1)
        sym_abs_cnt = sum(sym_dist < 25)
        print(f"idx={idx}, asym_num_pos={abs_cnt}, asym_frac_pos={100*(abs_cnt/all_c.shape[0])},\
              sym_num_pos={sym_abs_cnt}, sym_frac_pos={100 * (sym_abs_cnt/all_c.shape[0])}")
    return labels

def generate_embeddings_data(av):
    all_q = generate_queries(av)
    all_c = generate_corpus(av, all_q)
    T = get_synthT(all_q, all_c)
    #T1 = get_T1()

    all_data = {
        "all_q": all_q,
        "all_c": all_c,
        "T": T,
        "q_thresh": av.q_thresh
    }
    fp = "./data/SyntheticData_delta" + str(av.delta) + "_embed_dim"+ str(av.embed_dim) +\
        "_num_q" + str(av.num_q) + "_num_cpos_perq" + str(av.num_cpos_perq) + "_num_cneg_perq" +\
        str(av.num_cneg_perq) + "_T" + str(T) + ".pkl"
    pickle.dump(all_data, open(fp,"wb"))
    print(f"Saved data to {fp}")

def generate_labels(av):
    fp = "./data/SyntheticData_delta" + str(av.delta) + "_embed_dim"+ str(av.embed_dim) +\
        "_num_q" + str(av.num_q) + "_num_cpos_perq" + str(av.num_cpos_perq) + "_num_cneg_perq" +\
        str(av.num_cneg_perq) + "_T" + str(av.synthT) + ".pkl"
    assert os.path.exists(fp)
    logger.info("Fetching embeddings from %s", fp)
    embed_d = pickle.load(open(fp, "rb"))
    all_data = embed_d

    #T = get_T(all_data['all_q'], all_data['all_c']) ### comment after use
    #print("T=",T)
    labels = assign_labels(embed_d['all_q'], embed_d['all_c'], embed_d['T'], av.T1)
    
    #all_data['T'] = T ### comment after use
    all_data["T1"] = av.T1
    all_data["positive_labels"] = labels

    fp = "./data/SyntheticData_delta" + str(av.delta) + "_embed_dim"+ str(av.embed_dim) +\
        "_num_q" + str(av.num_q) + "_num_cpos_perq" + str(av.num_cpos_perq) + "_num_cneg_perq" +\
        str(av.num_cneg_perq) + "_T" + str(av.synthT) + "_T1" + str(av.T1) +".pkl"
    pickle.dump(all_data, open(fp, "wb"))
    logger.info("Dumping embeddings and ground truth labels into %s", fp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta",                   type=int,   default=15)
    ap.add_argument("--num_q",                   type=int,   default=200)
    ap.add_argument("--q_thresh",                type=int,   default=20)
    ap.add_argument("--pos_low",                 type=int,   default=4)
    ap.add_argument("--pos_high",                type=int,   default=8)
    ap.add_argument("--neg_low",                 type=int,   default=1)
    ap.add_argument("--neg_high",                type=int,   default=3)
    ap.add_argument("--num_c_perq",              type=int,   default=1000)
    ap.add_argument("--num_cpos_perq",           type=int,   default=100)
    ap.add_argument("--num_cneg_perq",           type=int,   default=900)
    ap.add_argument("--embed_dim",               type=int,   default=10)
    ap.add_argument("--T1",                      type=int,   default=1)
    ap.add_argument("--synthT",                  type=int,   default=133) 
    ap.add_argument("--save_embed",              action='store_true')
    ap.add_argument("--save_labels",             action='store_true')

    av = ap.parse_args()

    np.random.seed(0)
    if av.save_embed:
        generate_embeddings_data(av)
    elif av.save_labels:
        generate_labels(av)
    else:
        print("Please inform whether you want to save embeddings or labels!")
