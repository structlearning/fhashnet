import argparse
import sys
import os
import random
import gzip
import tqdm
import urllib.request
import csv
import time
import torch
from torch import nn, Tensor
import numpy as np
from datetime import datetime
import math
from sentence_transformers import SentenceTransformer, util,  InputExample, datasets, models, losses
from src.earlystopping import EarlyStoppingModule
from common import logger, set_log
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from pebble import ProcessPool
from typing import Iterable, Dict
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = False
import pickle
from collections import Counter


def issubset_multiset(X, Y, av):
    if av.CsubQ:
      return len(Counter(Y)-Counter(X)) == 0
    else:
      return len(Counter(X)-Counter(Y)) == 0

def issubset_set(X, Y, av):
  if av.CsubQ:
    return Y.issubset(X)
  else:
    return X.issubset(Y)

def wjac_sim(a: Tensor, b: Tensor, av):
    """
    input dim: a (n x d or 1 x d), b (n x d)
    output dim: n x 1
    Computes the weighted jaccard similarity  for all i and j.
    :return: Matrix with res[i][j]  = wieghtedJaccard(a[i]- b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    assert len(a.shape) == 2 and len(b.shape) == 2 and (b.shape[0] == a.shape[0] or a.shape[0] == 1)

    return torch.minimum(a,b).sum(axis=1)/(torch.maximum(a,b).sum(axis=1)+1e-8)

def hinge_sim(a: Tensor, b: Tensor, av):
    """
    input dim: a (n x d or 1 x d), b (n x d)
    output dim: n x 1
    Computes the asym hinge similarity -max(0,a[i]- b[j]) for all i and j.
    :return: Matrix with res[i][j]  = -max(0,a[i]- b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    assert len(a.shape) == 2 and len(b.shape) == 2 and (b.shape[0] == a.shape[0] or a.shape[0] == 1)

    if av.CsubQ:
      return -(nn.ReLU()(b-a)).sum(-1)
    else:
      return -(nn.ReLU()(a-b)).sum(-1)

def sigmoid_hinge_sim(model_obj, a, b, av):
    """
    input dim: a (n x d or 1 x d), b (n x d)
    output dim: n x 1
    Computes the asym hinge similarity -max(0,a[i]- b[j]) for all i and j.
    :return: Matrix with res[i][j]  = -max(0,a[i]- b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    assert len(a.shape) == 2 and len(b.shape) == 2 and (b.shape[0] == a.shape[0] or a.shape[0] == 1)

    if av.CsubQ:
      sig_in= (nn.ReLU()(b-a)).sum(-1)
    else:
      sig_in= (nn.ReLU()(a-b)).sum(-1)

    return torch.nn.Sigmoid()(model_obj.sigmoid_a*sig_in + model_obj.sigmoid_b)

def dot_sim(a: Tensor, b: Tensor):
    """
    input dim: a (n x d or 1 x d), b (n x d)
    output dim: n x 1
    Computes the dot product similarity a[i].b[i] for all i
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    assert len(a.shape) == 2 and len(b.shape) == 2 and (b.shape[0] == a.shape[0] or a.shape[0] == 1)

    return (a * b).sum(-1)

def fetch_all_msweb_data(fname):
    if not os.path.exists(fname):
        urllib.request.urlretrieve("https://kdd.ics.uci.edu/databases/msweb/anonymous-msweb.data.gz", fname)
    fIn =  gzip.open(fname, 'rt', encoding='utf8')
    elements = []
    corpus = []
    curr_corpus_item = set()
    for l in fIn:
        # print(l[0], l)
        if l[0] == 'A':
            pieces = l.split(",")
            elements.append({"id":int(pieces[1]), "text": pieces[3].strip('"'), "url": pieces[4].strip('/"')})
        if l[0] == 'C':
            corpus.append(curr_corpus_item)
            curr_corpus_item = set()
        if l[0] == 'V':
            curr_corpus_item.add(int(l.split(",")[1]))
    
    unique_corpus = []
    for i,c1 in tqdm.tqdm(enumerate(corpus)):
        isUnique = True
        for j in range(i+1, len(corpus)):
            if c1 == corpus[j]:
                isUnique = False
        if isUnique:
            unique_corpus.append(c1)
    logger.info(f"#elements = {len(elements)}, Non-unique dataset size = {len(corpus)}, Unique dataset size = {len(unique_corpus)}")

    return elements, unique_corpus

def fetch_all_msnbc_data(fname):
    if not os.path.exists(fname):
        urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/msnbc-mld/msnbc990928.seq.gz", fname)
    fIn =  gzip.open(fname, 'rt', encoding='utf8')
    elements = []
    corpus = []
    stringified_corpus = []
    curr_corpus_item = list()
    for l in tqdm.tqdm(fIn):
        l = l.strip()
        if  l == '' or l[0] == '%':
            continue
        tokens = l.split(" ")
        if tokens[0].isnumeric():
            for t in tokens:
                curr_corpus_item.append(int(t))
            stringified = "@".join([str(e) for e in sorted(curr_corpus_item)])
            if stringified not in stringified_corpus:
                corpus.append(curr_corpus_item)
                stringified_corpus.append(stringified)
            curr_corpus_item = list()
        else:
            elements = [{"id":i+1, "text": t} for i,t in enumerate(tokens)]
    filtered_corpus = [c for c in corpus if len(c) <= 50]
    # unique_corpus = []
    # for i,c1 in tqdm.tqdm(enumerate(corpus)):
    #     isUnique = True
    #     for j in range(i+1, len(corpus)):
    #         if c1 == corpus[j]:
    #             isUnique = False
    #     if isUnique:
    #         unique_corpus.append(c1)
    # logger.info(f"#elements = {len(elements)}, Non-unique dataset size = {len(corpus)}, Unique dataset size = {len(unique_corpus)}")

    return elements, filtered_corpus

def load_all_msweb_data(fname, av):
  fname2 = os.path.join("data", "anonymous-msweb.data.gz")
  fname3 = os.path.join("data", "msnbc990928.seq.gz")
  if not os.path.exists(fname) and av.DATASET_NAME == "MSWEB":
    logger.info(f"Creating msweb-data-processed file, as file not found with fname {fname}")
    elements, corpus = fetch_all_msweb_data(fname2)
    data = {'elements': elements, 'corpus': corpus}
    pickle.dump(data, open(fname, "wb"))
  elif not os.path.exists(fname) and av.DATASET_NAME == "MSNBC":
    logger.info(f"Creating msnbc-data-processed file, as file not found with fname {fname}")
    elements, corpus = fetch_all_msnbc_data(fname3)
    data = {'elements': elements, 'corpus': corpus}
    pickle.dump(data, open(fname, "wb"))
  data = pickle.load(open(fname, "rb"))

  return data['elements'], data['corpus']

def fetch_query_ids(unique_corpus, av, a=8, b=560):
    cntQueries = 0
    query_ids = []
    cntCorpus = [0 for _ in range(len(unique_corpus))]
    if av.DATASET_NAME == 'MSWEB'  and av.SKEW==0:
      for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
          for c2 in unique_corpus:
              if c1 != c2 and issubset_set(c1, c2, av):
                  cntCorpus[i] += 1
          logger.info(f"{cntCorpus[i]}, {cntQueries}")
          if cntCorpus[i]  > a and cntCorpus[i]  < b:
                cntQueries += 1
                query_ids.append(i)
          if cntQueries >= av.NUM_QUERIES:
              break
    elif av.DATASET_NAME == 'MSWEB'  and av.SKEW==1:
      a = 8
      b = 50
      for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
          for c2 in unique_corpus:
              if c1 != c2 and issubset_set(c1, c2, av):
                  cntCorpus[i] += 1
          logger.info(f"{cntCorpus[i]}, {cntQueries}")
          if cntCorpus[i]  > a and cntCorpus[i]  < b:
                cntQueries += 1
                query_ids.append(i)
          if cntQueries >= av.NUM_QUERIES:
              break
    elif av.DATASET_NAME == 'MSNBC' and av.SKEW==0:
      a = 8
      b = 7000
      fname = os.path.join("data", f"msnbc_{'csubq_' if av.CsubQ else ''}query_ids_first{av.NUM_QUERIES}.pkl")
      if not os.path.exists(fname):
        for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_multiset(c1, c2, av):
                    cntCorpus[i] += 1
            logger.info(f"{cntCorpus[i]}, {cntQueries}")
            if cntCorpus[i]  > 8 and cntCorpus[i]  < 7000:
                cntQueries += 1
                query_ids.append(i)
            if cntQueries >= av.NUM_QUERIES:
                break
        all_d = {"query_ids": query_ids}
        pickle.dump(all_d, open(fname, "wb"))
      all_d = pickle.load(open(fname, "rb"))
      query_ids = all_d['query_ids']
    elif av.DATASET_NAME == 'MSNBC' and av.SKEW==3:
      a = 8
      b = 45
      fname = os.path.join("data", f"msnbc_{'csubq_' if av.CsubQ else ''}query_ids_first{av.NUM_QUERIES}_SKEW{av.SKEW}.pkl")
      if not os.path.exists(fname):
        for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_multiset(c1, c2, av):
                    cntCorpus[i] += 1
            logger.info(f"{cntCorpus[i]}, {cntQueries}")
            if cntCorpus[i]  > a and cntCorpus[i]  < b:
                cntQueries += 1
                query_ids.append(i)
            if cntQueries >= av.NUM_QUERIES:
                break
        all_d = {"query_ids": query_ids}
        pickle.dump(all_d, open(fname, "wb"))
      all_d = pickle.load(open(fname, "rb"))
      query_ids = all_d['query_ids']
    elif av.DATASET_NAME == 'MSNBC' and av.SKEW==4:
      a = 8
      b = 35
      fname = os.path.join("data", f"msnbc_{'csubq_' if av.CsubQ else ''}query_ids_first{av.NUM_QUERIES}_SKEW{av.SKEW}.pkl")
      if not os.path.exists(fname):
        for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_multiset(c1, c2, av):
                    cntCorpus[i] += 1
            logger.info(f"{cntCorpus[i]}, {cntQueries}")
            if cntCorpus[i]  > a and cntCorpus[i]  < b:
                cntQueries += 1
                query_ids.append(i)
            if cntQueries >= av.NUM_QUERIES:
                break
        all_d = {"query_ids": query_ids}
        pickle.dump(all_d, open(fname, "wb"))
      all_d = pickle.load(open(fname, "rb"))
      query_ids = all_d['query_ids']
    elif av.DATASET_NAME == 'MSNBC' and av.SKEW==5:
      a = 8
      b = 25
      fname = os.path.join("data", f"msnbc_{'csubq_' if av.CsubQ else ''}query_ids_first{av.NUM_QUERIES}_SKEW{av.SKEW}.pkl")
      if not os.path.exists(fname):
        for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_multiset(c1, c2, av):
                    cntCorpus[i] += 1
            logger.info(f"{cntCorpus[i]}, {cntQueries}")
            if cntCorpus[i]  > a and cntCorpus[i]  < b:
                cntQueries += 1
                query_ids.append(i)
            if cntQueries >= av.NUM_QUERIES:
                break
        all_d = {"query_ids": query_ids}
        pickle.dump(all_d, open(fname, "wb"))
      all_d = pickle.load(open(fname, "rb"))
      query_ids = all_d['query_ids']
    elif av.DATASET_NAME == 'MSNBC' and av.SKEW==6:
      a = 8
      b = 15
      fname = os.path.join("data", f"msnbc_{'csubq_' if av.CsubQ else ''}query_ids_first{av.NUM_QUERIES}_SKEW{av.SKEW}.pkl")
      if not os.path.exists(fname):
        for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_multiset(c1, c2, av):
                    cntCorpus[i] += 1
            logger.info(f"{cntCorpus[i]}, {cntQueries}")
            if cntCorpus[i]  > a and cntCorpus[i]  < b:
                cntQueries += 1
                query_ids.append(i)
            if cntQueries >= av.NUM_QUERIES:
                break
        all_d = {"query_ids": query_ids}
        pickle.dump(all_d, open(fname, "wb"))
      all_d = pickle.load(open(fname, "rb"))
      query_ids = all_d['query_ids']
    elif av.DATASET_NAME == 'MSNBC' and av.SKEW==7:
      a = 3
      b = 14
      fname = os.path.join("data", f"msnbc_{'csubq_' if av.CsubQ else ''}query_ids_first{av.NUM_QUERIES}_SKEW{av.SKEW}.pkl")
      if not os.path.exists(fname):
        for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_multiset(c1, c2, av):
                    cntCorpus[i] += 1
            logger.info(f"{cntCorpus[i]}, {cntQueries}")
            if cntCorpus[i]  > a and cntCorpus[i]  < b:
                cntQueries += 1
                query_ids.append(i)
            if cntQueries >= av.NUM_QUERIES:
                break
        all_d = {"query_ids": query_ids}
        pickle.dump(all_d, open(fname, "wb"))
      all_d = pickle.load(open(fname, "rb"))
      query_ids = all_d['query_ids']
    elif av.DATASET_NAME == 'MSNBC' and av.SKEW==1:
      fname = os.path.join("data", f"msnbc_{'csubq_' if av.CsubQ else ''}query_ids_first{av.NUM_QUERIES}_SKEW{av.SKEW}.pkl")
      if not os.path.exists(fname):
        cntQueries1 = 0
        query_ids1 = []
        cntQueries2 = 0
        query_ids2= []
        for i,c1 in tqdm.tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_multiset(c1, c2, av):
                    cntCorpus[i] += 1
            logger.info(f"{cntCorpus[i]}, {cntQueries}")
            if cntCorpus[i]  > 8 and cntCorpus[i]  < 75:
                cntQueries1 += 1
                query_ids1.append(i)
            if (cntQueries1 >200) and (cntQueries2 <300)  and cntCorpus[i]  > 8 and cntCorpus[i]  < 560:
                cntQueries2 += 1
                query_ids2.append(i)
            if cntQueries1 >= av.NUM_QUERIES and cntQueries2 >= 300:
                break
        all_d = {"query_ids": [query_ids1,query_ids2]}
        pickle.dump(all_d, open(fname, "wb"))
      all_d = pickle.load(open(fname, "rb"))
      query_ids = all_d['query_ids']
    else:
      raise NotImplementedError()

    logger.info(f"Found {len(query_ids)} queries which have positive corpus items b/w {a} and {b}.")
    return query_ids

def remove_query_sets_from_corpus(unique_corpus, query_ids_list):
  queries = [[unique_corpus[qid] for qid in query_ids] for query_ids in query_ids_list]
  #print(queries)  
  comb_q = set().union(*list(map(set,query_ids_list)))
  disjoint_corpus = [unique_corpus[cid] for cid in range(len(unique_corpus)) if cid not in comb_q]

  return queries, disjoint_corpus

def remove_queries_from_corpus(unique_corpus, query_ids):
  queries = [unique_corpus[qid] for qid in query_ids]
  disjoint_corpus = [unique_corpus[cid] for cid in range(len(unique_corpus)) if cid not in query_ids]

  return queries, disjoint_corpus

def train(av):
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    # elements, unique_corpus = fetch_all_msweb_data(os.path.join("data", "anonymous-msweb.data.gz"))
    elements, unique_corpus = load_all_msweb_data(os.path.join("data", f"{av.DATASET_NAME.lower()}-processed.pkl"), av)
    query_ids = fetch_query_ids(unique_corpus, av)
    if av.DATASET_NAME == 'MSNBC' and av.SKEW==1:
        queries, unique_corpus =  remove_query_sets_from_corpus(unique_corpus, query_ids)
        queries = queries[0]
    else:
        query_ids = query_ids[:av.NUM_QUERIES]
        queries, unique_corpus = remove_queries_from_corpus(unique_corpus, query_ids)
    # random.shuffle(query_ids)
    # val_split = int(av.VAL_FRAC * len(query_ids))
    # test_split = int(av.TEST_FRAC * len(query_ids))
    val_queries, tr_queries, _ = queries[:av.MAX_QUERIES], queries[av.MAX_QUERIES:2 * av.MAX_QUERIES], queries[2 * av.MAX_QUERIES:]
    
    logger.info(f"Final corpus size: {len(unique_corpus)}")
    logger.info(f"Total num queries: {len(queries)}")
    logger.info(f" - Train num queries: {len(tr_queries)}")
    logger.info(f" - Val num queries: {len(val_queries)}")
    logger.info(f" - Test num queries: {len(queries) - len(tr_queries) - len(val_queries)}")


    model_name ='distilroberta-base'
    word_embedding_model = models.Transformer(model_name, max_seq_length=5).to(device)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean').to(device)
    sent_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device)
    logger.info(sent_model)

    train_data = OurMSwebData(av, elements, unique_corpus, tr_queries, sent_model)
    val_data = OurMSwebData(av, elements, unique_corpus, val_queries, sent_model)
    
    if av.DEEPSET_VER == "Identity": 
        deepset_model = DeepSetVarSize0(train_data.query_tensors.shape[2]).to(device) 
    elif av.DEEPSET_VER == "LinWoBias":
        deepset_model = DeepSetVarSize1(train_data.query_tensors.shape[2]).to(device) 
    elif av.DEEPSET_VER == "LRLWoBias":
        deepset_model = DeepSetVarSize2(train_data.query_tensors.shape[2]).to(device) 
    elif av.DEEPSET_VER == "LRLRLWoBias":
        deepset_model = DeepSetVarSize3(train_data.query_tensors.shape[2]).to(device) 
    elif av.DEEPSET_VER.startswith("Default"):
        deepset_model = DeepSetVarSize(word_embedding_model.get_word_embedding_dimension(), av.IN_D, av.HIDDEN_LAYERS, av, dev=device)
    elif av.DEEPSET_VER == "UA":
        deepset_model = DeepSetVarSize5(train_data.query_tensors.shape[2], av.IN_HIDDEN_LAYERS, av.OUT_HIDDEN_LAYERS, av, dev=device)
    else:
        deepset_model = DeepSetVarSize4(train_data.query_tensors.shape[2], av.DEEPSET_VER, dev=device)
    
    optimizer = torch.optim.Adam(deepset_model.parameters(),
                                    lr=av.LEARNING_RATE,
                                    weight_decay=av.WEIGHT_DECAY)
    cnt = 0
    for param in deepset_model.parameters():
            cnt=cnt+torch.numel(param)
    logger.info("no. of params in model: %s",cnt)
    es = EarlyStoppingModule(av,av.ES)

    losses = []
    epoch_changes = []
    if not os.path.exists("plotDir"):
        logger.info("Did not find plotDir, so creating one")
        os.makedirs("plotDir")
    run = 0
    while True:
        loss_plot_path = os.path.join("plotDir", f"tr_loss_{av.TASK}_RUN{run}.png")
        if os.path.exists(loss_plot_path):
            run += 1
        else:
            logger.info(f"Saving training loss plot in file: {loss_plot_path}")
            break
    run = 0
    while av.RUN_TILL_ES or run < av.NUM_RUNS:
        create_batches_start = time.time()
        if av.P2N > 0:
          n_batches = train_data.create_batches_with_p2n()
        else:
          n_batches = train_data.create_stratified_batches()
        create_batches_time = time.time() - create_batches_start
        logger.info(f"# Batches = {n_batches}, Time: {create_batches_time}")
        epoch_loss = 0
        start_time = time.time()
        for i in range(n_batches):
            start_time_batch = time.time()
            if av.P2N > 0:
              batch_corpus_tensors, batch_corpus_mask_tensors, batch_query_tensors, batch_query_mask_tensors, batch_target, batch_query_ids = train_data.fetch_batched_data_by_id_optimized(i)
            else:
              batch_corpus_tensors, batch_corpus_mask_tensors, batch_query_tensors, batch_query_mask_tensors, batch_target, batch_query_ids = train_data.fetch_batched_data_by_id(i)
            batch_loading_time = time.time() - start_time_batch

            optimizer.zero_grad()
            
            start_forward_pass = time.time()
            batch_corpus_embeds = deepset_model(batch_corpus_tensors,batch_corpus_mask_tensors)
            batch_query_embeds = deepset_model(batch_query_tensors,batch_query_mask_tensors)
            forward_pass_time = time.time() - start_forward_pass

            start_scoring = time.time()
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            if av.TASK.startswith("SymScore"):
                prediction = cos(batch_query_embeds, batch_corpus_embeds)
            elif av.TASK.startswith("AsymScore"):
                prediction = hinge_sim(batch_query_embeds, batch_corpus_embeds, av)
            elif av.TASK.startswith("WeightedJaccard"):
                prediction = wjac_sim(batch_query_embeds, batch_corpus_embeds, av)
            elif av.TASK.startswith("SigmoidAsymScore"):
                prediction = sigmoid_hinge_sim(deepset_model,batch_query_embeds, batch_corpus_embeds, av)
            elif av.TASK.startswith("DotProduct"):
                prediction = dot_sim(batch_query_embeds, batch_corpus_embeds)
            else:
                raise NotImplementedError()
            scoring_time = time.time() - start_scoring

            #Pairwise ranking loss
            predPos = prediction[batch_target>0.5]
            predNeg = prediction[batch_target<0.5]
           
            bceloss = torch.nn.BCEWithLogitsLoss()
 
            start_loss_com = time.time()
            if av.TASK.startswith("SigmoidAsymScore") or av.USE_SIG:
              loss = bceloss(prediction, batch_target.cuda())
            elif av.PER_QUERY:
              qidPos = batch_query_ids[batch_target>0.5]
              qidNeg = batch_query_ids[batch_target<0.5]
              loss, frac_pairs_used_2_train = pairwise_ranking_loss_similarity_per_query(predPos.unsqueeze(1),predNeg.unsqueeze(1),qidPos.unsqueeze(1),qidNeg.unsqueeze(1), av)
              if i % av.WRITE_EVERY == 0:
                print(f"frac_pairs_used_2_train={frac_pairs_used_2_train}")
            else:  
              loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), av)
            loss_com_time = time.time() - start_loss_com
            #losses = torch.nn.functional.mse_loss(target, prediction,reduction="sum")

            start_backprop = time.time()
            loss.backward()
            optimizer.step()
            backprop_time = time.time() - start_backprop
            epoch_loss = epoch_loss + loss.item()

            losses.append(float(loss.item()))
            start_plot_time = time.time()
            if i % av.WRITE_EVERY == 0:
              plt.figure()
              plt.title("Loss vs GD Steps")
              plt.xlabel("GD Steps")
              plt.ylabel("Loss")
              plt.plot(range(len(losses)), losses, '-D', markevery=epoch_changes)
              plt.savefig(loss_plot_path)
              plt.close()
              plot_time = time.time() - start_plot_time
              print(f"Batch loading: {batch_loading_time}, Forward Pass: {forward_pass_time}, Scoring: {scoring_time}, Loss Com: {loss_com_time}, Backprop: {backprop_time}, Plot: {plot_time}, Total: {time.time() - start_time_batch}")
              print("Epoch #: {:d}, Batch #: {:d} / {:d}, train loss: {:f} Time: {:.2f}".format(run,i,n_batches,loss.item(),time.time()-start_time_batch))

        logger.info("\n\nEnd of Epoch #: %d train loss: %f Time: %.2f",run,epoch_loss,time.time()-start_time)
    
        start_time = time.time()
        tr_ap_score,tr_all_ap,tr_map_score = evaluate_embeddings_similarity(av,deepset_model,train_data)
        logger.info("Run: %d TRAIN ap_score: %.6f map_score: %.6f Time: %.2f",run,tr_ap_score,tr_map_score,time.time()-start_time)
        start_time = time.time()
        ap_score,all_ap,map_score = evaluate_embeddings_similarity(av,deepset_model,val_data)
        logger.info("Run: %d VAL ap_score: %.6f map_score: %.6f Time: %.2f",run,ap_score,map_score,time.time()-start_time)
        if av.RUN_TILL_ES:
            if es.check([map_score],deepset_model,run):
                break
        epoch_changes.append(len(losses))
        run += 1
def test(av):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  elements, unique_corpus = load_all_msweb_data(os.path.join("data", f"{av.DATASET_NAME.lower()}-processed.pkl"), av)
  query_ids = fetch_query_ids(unique_corpus, av)
  query_ids = query_ids[:av.NUM_QUERIES]
  queries, unique_corpus = remove_queries_from_corpus(unique_corpus, query_ids)
  val_queries, tr_queries, test_queries = queries[:av.MAX_QUERIES], queries[av.MAX_QUERIES:2 * av.MAX_QUERIES], queries[2 * av.MAX_QUERIES:]
  
  logger.info(f"Final corpus size: {len(unique_corpus)}")
  logger.info(f"Total num queries: {len(queries)}")
  logger.info(f" - Train num queries: {len(tr_queries)}")
  logger.info(f" - Val num queries: {len(val_queries)}")
  logger.info(f" - Test num queries: {len(test_queries)}")


  model_name ='distilroberta-base'
  word_embedding_model = models.Transformer(model_name, max_seq_length=5)
  pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
  sent_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
  logger.info(sent_model)

  train_data = OurMSwebData(av, elements, unique_corpus, tr_queries, sent_model)
  val_data = OurMSwebData(av, elements, unique_corpus, val_queries, sent_model)
  test_data = OurMSwebData(av, elements, unique_corpus, test_queries, sent_model)

  if av.DEEPSET_VER == "Identity": 
      deepset_model = DeepSetVarSize0(train_data.query_tensors.shape[2]).to(device) 
  elif av.DEEPSET_VER == "LinWoBias":
      deepset_model = DeepSetVarSize1(train_data.query_tensors.shape[2]).to(device) 
  elif av.DEEPSET_VER == "LRLWoBias":
      deepset_model = DeepSetVarSize2(train_data.query_tensors.shape[2]).to(device) 
  elif av.DEEPSET_VER == "LRLRLWoBias":
      deepset_model = DeepSetVarSize3(train_data.query_tensors.shape[2]).to(device) 
  elif av.DEEPSET_VER.startswith("Default"):
      deepset_model = DeepSetVarSize(word_embedding_model.get_word_embedding_dimension(), av.IN_D, av.HIDDEN_LAYERS, av, dev=device)
  elif av.DEEPSET_VER == "UA":
      deepset_model = DeepSetVarSize5(train_data.query_tensors.shape[2], av.IN_HIDDEN_LAYERS, av.OUT_HIDDEN_LAYERS, av, dev=device)
  else:
      deepset_model = DeepSetVarSize4(train_data.query_tensors.shape[2], av.DEEPSET_VER, dev=device)

  es = EarlyStoppingModule(av,av.ES)
  checkpoint = es.load_best_model()
  deepset_model.load_state_dict(checkpoint['model_state_dict'])

  start_time = time.time()
  te_ap_score,te_all_ap,te_map_score = evaluate_embeddings_similarity(av,deepset_model,test_data)
  logger.info("TEST ap_score: %.6f map_score: %.6f Time: %.2f",te_ap_score,te_map_score,time.time()-start_time)
  start_time = time.time()
  ap_score,all_ap,map_score = evaluate_embeddings_similarity(av,deepset_model,val_data)
  logger.info("VAL ap_score: %.6f map_score: %.6f Time: %.2f",ap_score,map_score,time.time()-start_time)
  start_time = time.time()
  ap_score,all_ap,map_score = evaluate_embeddings_similarity(av,deepset_model,train_data)
  logger.info("TRAIN ap_score: %.6f map_score: %.6f Time: %.2f",ap_score,map_score,time.time()-start_time)


def evaluate_embeddings_similarity(av,deepset_model,sampler):
  deepset_model.eval()
  d_pos = sampler.list_pos
  d_neg = sampler.list_neg
  d = d_pos + d_neg
  npos = len(d_pos)
  nneg = len(d_neg)

  pred = []

  n_batches = sampler.create_batches(d, "ap")
  for i in tqdm.tqdm(range(n_batches)):
    #ignoring target values and qids here since not needed for AP ranking score 
    batch_corpus_tensors, batch_corpus_mask_tensors, batch_query_tensors, batch_query_mask_tensors, _, _ = sampler.fetch_batched_data_by_id_optimized(i)
    batch_corpus_embeds = deepset_model(batch_corpus_tensors,batch_corpus_mask_tensors)
    batch_query_embeds = deepset_model(batch_query_tensors,batch_query_mask_tensors)
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if av.TASK.startswith("SymScore"):
        prediction = cos(batch_query_embeds, batch_corpus_embeds)
    elif av.TASK.startswith("AsymScore"):
        prediction = hinge_sim(batch_query_embeds, batch_corpus_embeds, av)
    elif av.TASK.startswith("WeightedJaccard"):
        prediction = wjac_sim(batch_query_embeds, batch_corpus_embeds, av)
    elif av.TASK.startswith("SigmoidAsymScore"):
        prediction = sigmoid_hinge_sim(deepset_model,batch_query_embeds, batch_corpus_embeds, av)
    elif av.TASK.startswith("DotProduct"):
        prediction = dot_sim(batch_query_embeds, batch_corpus_embeds)
    else:
        raise NotImplementedError()
    
    pred.append( prediction.data )

  all_pred = torch.cat(pred,dim=0) 
  labels = torch.cat((torch.ones(npos),torch.zeros(nneg)))
  ap_score = average_precision_score(labels.cpu(), all_pred.cpu())   
  
  # MAP computation
  all_ap = []
  pred = []
  d = sampler.list_total_arranged_per_query
  n_batches = sampler.create_batches(d, "map")
  for i in tqdm.tqdm(range(n_batches)):
    #ignoring target values and qids here since not needed for AP ranking score 
    batch_corpus_tensors, batch_corpus_mask_tensors, batch_query_tensors, batch_query_mask_tensors, _, _ = sampler.fetch_batched_data_by_id_optimized(i)
    batch_corpus_embeds = deepset_model(batch_corpus_tensors,batch_corpus_mask_tensors)
    batch_query_embeds = deepset_model(batch_query_tensors,batch_query_mask_tensors)
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if av.TASK.startswith("SymScore"):
        prediction = cos(batch_query_embeds, batch_corpus_embeds)
    elif av.TASK.startswith("AsymScore"):
        prediction = hinge_sim(batch_query_embeds, batch_corpus_embeds, av)
    elif av.TASK.startswith("WeightedJaccard"):
        prediction = wjac_sim(batch_query_embeds, batch_corpus_embeds, av)
    elif av.TASK.startswith("SigmoidAsymScore"):
        prediction = sigmoid_hinge_sim(deepset_model,batch_query_embeds, batch_corpus_embeds, av)
    elif av.TASK.startswith("DotProduct"):
        prediction = dot_sim(batch_query_embeds, batch_corpus_embeds)
    else:
        raise NotImplementedError()
    
    pred.append( prediction.data )

  all_pred = torch.cat(pred,dim=0)
  labels = sampler.labels_total_arranged_per_query
  corpus_size = sampler.corpus_size
  
  for q_id in tqdm.tqdm(range(len(sampler.queries))):
    q_pred = all_pred[q_id * corpus_size : (q_id+1) * corpus_size]
    q_labels = labels[q_id * corpus_size : (q_id+1) * corpus_size]
    ap = average_precision_score(q_labels, q_pred.cpu())
    all_ap.append(ap)
  # for q_id in tqdm.tqdm(range(len(sampler.queries))):
  #   dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
  #   dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
  #   npos = len(dpos)
  #   nneg = len(dneg)
  #   d = dpos+dneg
  #   if npos>0 and nneg>0:    
  #     n_batches = sampler.create_batches(d) 
  #     pred = []  
  #     for i in range(n_batches):
  #       batch_corpus_tensors, batch_corpus_set_sizes, batch_query_tensors, batch_query_set_sizes, _ = sampler.fetch_batched_data_by_id_optimized(i)
  #       batch_corpus_embeds = deepset_model(batch_corpus_tensors,batch_corpus_set_sizes)
  #       batch_query_embeds = deepset_model(batch_query_tensors,batch_query_set_sizes)
        
  #       cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  #       if av.TASK.startswith("SymScore"):
  #           prediction = cos(batch_query_embeds, batch_corpus_embeds)
  #       elif av.TASK.startswith("AsymScore"):
  #           prediction = hinge_sim(batch_query_embeds, batch_corpus_embeds)
  #       elif av.TASK.startswith("DotProduct"):
  #           prediction = dot_sim(batch_query_embeds, batch_corpus_embeds)
  #       else:
  #           raise NotImplementedError()
        
  #       pred.append( prediction.data)
      
  #     all_pred = torch.cat(pred,dim=0) 
  #     labels = torch.cat((torch.ones(npos),torch.zeros(nneg)))
  #     ap = average_precision_score(labels.cpu(), all_pred.cpu()) 
  #     all_ap.append(ap)  
  return ap_score, all_ap, np.mean(all_ap)

class DeepSetVarSize0(torch.nn.Module):
    def  __init__(self,num_feature): 
        super(DeepSetVarSize0, self).__init__()
        self.num_feature = num_feature
        self.lin = torch.nn.Linear(self.num_feature,self.num_feature,bias=False)
        self.lin.weight = torch.nn.Parameter(torch.eye(self.num_feature))
    
    def forward (self,x,set_sizes=None):
        return self.lin(x).sum(-2)
    
class DeepSetVarSize1(torch.nn.Module):
    def  __init__(self,num_feature): 
        super(DeepSetVarSize1, self).__init__()
        self.num_feature = num_feature
        self.lin = torch.nn.Linear(self.num_feature,self.num_feature,bias=False)
        self.lin.weight = torch.nn.Parameter(torch.exp(self.lin.weight))
    
    def forward (self,x,set_sizes=None):
        return self.lin(x).sum(-2)

        
class DeepSetVarSize2(torch.nn.Module):
    def  __init__(self,num_feature): 
        super(DeepSetVarSize2, self).__init__()
        self.num_feature = num_feature
        self.lin1 = torch.nn.Linear(self.num_feature,self.num_feature,bias=False)
        self.lin2 = torch.nn.Linear(self.num_feature,self.num_feature,bias=False)
        self.lin1.weight = torch.nn.Parameter(torch.exp(self.lin1.weight))
        self.lin2.weight = torch.nn.Parameter(torch.exp(self.lin2.weight))
        self.relu1 = torch.nn.ReLU()
        
    def forward (self,x,set_sizes=None):
        return self.lin2(self.relu1(self.lin1(x))).sum(-2)

class DeepSetVarSize3(torch.nn.Module):
    def  __init__(self,num_feature): 
        super(DeepSetVarSize3, self).__init__()
        self.num_feature = num_feature
        self.lin1 = torch.nn.Linear(self.num_feature,self.num_feature,bias=False)
        self.lin2 = torch.nn.Linear(self.num_feature,self.num_feature,bias=False)
        self.lin3 = torch.nn.Linear(self.num_feature,self.num_feature,bias=False)
        self.lin1.weight = torch.nn.Parameter(torch.exp(self.lin1.weight))
        self.lin2.weight = torch.nn.Parameter(torch.exp(self.lin2.weight))
        self.lin3.weight = torch.nn.Parameter(torch.exp(self.lin3.weight))
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        
    def forward (self,x,set_sizes=None):
        return self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(x))))).sum(-2)

class DeepSetVarSize4(torch.nn.Module):
    def  __init__(self,num_feature,layers="LRL",dev='cuda'): 
        super(DeepSetVarSize4, self).__init__()
        self.num_feature = num_feature
        self.net = []
        for l in layers:
          if l == 'L':
            self.net.append(torch.nn.Linear(self.num_feature,self.num_feature,bias=False))
          elif l == 'R':
            self.net.append(torch.nn.ReLU())
          else:
            raise NotImplementedError()
        
        self.net = torch.nn.Sequential(*self.net).to(dev)
    def forward (self,x,set_sizes=None):
        return self.net(x).sum(-2)

class DeepSetVarSize5(torch.nn.Module):
    def __init__(self, num_feature, in_hidden_layers, out_hidden_layers, av, dev='cuda'):
        super(DeepSetVarSize5, self).__init__()

        self.init_net = []
        #self.in_d = in_d
        inner_hs = [num_feature] + in_hidden_layers #+ [1]
        for h0, h1 in zip(inner_hs, inner_hs[1:]):
            self.init_net.append(torch.nn.Linear(h0, h1))
            if av.LAYER_NORMALIZE:
              self.init_net.append(torch.nn.LayerNorm(h1))
            self.init_net.append(torch.nn.ReLU())

        if not av.IN_RELU and not self.init_net == []:
          logger.info("Inner net: Popping the last ReLU")
          self.init_net.pop()  # pop the last relu
        elif self.init_net == []:
          logger.info("Inner net: No Relu to pop!")
        else:
          logger.info("Inner net: Not popping the last ReLU")

        #self.num_feature = num_feature
        self.net = []
        #self.in_d = in_d
        outer_hs = [inner_hs[-1]] + out_hidden_layers #+ [1]
        for h0, h1 in zip(outer_hs, outer_hs[1:]):
            self.net.append(torch.nn.Linear(h0, h1))
            if av.LAYER_NORMALIZE:
              self.net.append(torch.nn.LayerNorm(h1))
            self.net.append(torch.nn.ReLU())

        if not av.OUT_RELU and not self.net == []:
          logger.info("Out net: Popping the last ReLU")
          self.net.pop()  # pop the last relu
        elif self.net == []:
          logger.info("Out net: No Relu to pop!")
        else:
          logger.info("Out net: Not popping the last ReLU")

#         self.net.append(torch.nn.ELU())
        self.init_net = torch.nn.Sequential(*self.init_net).to(dev)
        self.net = torch.nn.Sequential(*self.net).to(dev)
        self.device = dev
        self.av = av
        if self.av.TASK.startswith("SigmoidAsymScore"):
            self.sigmoid_a = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
            self.sigmoid_b = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))

    def forward(self, x, mask):#, set_sizes=None):
        """
        mask is a tensor of 0s and 1s
        """
        if len(list(x.shape)) == 2:
            x = x.unsqueeze(0)
            # set_sizes = [x.shape[1]]
        #we expect batch_size*max_set_size*input_dim
        assert len(x.shape)==3
        #each set in the batch should have the corr. set size
        assert(x.shape[0]==mask.shape[0])
        # mask_start = time.time()
        # mask_ids = torch.arange(x.size(1)).unsqueeze(0) < torch.FloatTensor(set_sizes).unsqueeze(-1)
        # mask_end = time.time()
        # start = time.time()
        x = x.to(self.device)
        # gpu_transfer_end = time.time()
        x  = self.init_net(x)
        # inner_net_pass_end = time.time()
        assert x.shape == mask.shape
        # mask = torch.ones(x.shape, device=self.device)
        # mask_tensor_init_end = time.time()
        # mask[~mask_ids] = 0  
        # mask_tensor_end = time.time()
        # logger.info(f"x.shape = {x.shape}")
        x = x * mask
        # mult_end = time.time()
        # logger.info(f"x.shape = {x.shape}")
        x = torch.sum(x,dim=1)
        # sum_end = time.time()
        x  = self.net(x)
        # outer_net_end = time.time()
        # print(f"gpu_transfer={gpu_transfer_end-start}, inner_pass={inner_net_pass_end-gpu_transfer_end}, sum_mult={sum_end-inner_net_pass_end}, outer_pass={outer_net_end-sum_end}, total={outer_net_end-start}")
        return x

class DeepSetVarSize(torch.nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers, av, dev='cuda'):
        super(DeepSetVarSize, self).__init__()
       
        self.init_layer = torch.nn.Linear(num_feature,in_d).to(dev)
        self.num_feature = num_feature
        self.net = []
        self.in_d = in_d
        hs = [in_d] + hidden_layers #+ [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                torch.nn.Linear(h0, h1),
                torch.nn.ReLU(),
            ])
        if not av.DEEPSET_VER.endswith('WithReLU'):
          logger.info("Popping the last ReLU")
          self.net.pop()  # pop the last relu
        else:
          logger.info("Not popping the last ReLU")
#         self.net.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*self.net).to(dev)
        self.device = dev
         
         
    def forward(self, x, set_sizes=None):
        """
            set_sizes is list/nparray/1-D tensor of set sizes of length batch
        """
        if len(list(x.shape)) == 2:
            x = x.unsqueeze(0)
            set_sizes = [x.shape[1]]
        #we expect batch_size*max_set_size*input_dim
        assert len(x.shape)==3
        #each set in the batch should have the corr. set size
        assert(x.shape[0]==len(set_sizes))
        mask = torch.arange(x.size(1)).unsqueeze(0).to(self.device) < torch.FloatTensor(set_sizes).unsqueeze(-1)
        x = x.to(self.device)
        x  = self.init_layer(x)
        x[~mask] = 0
        x = torch.sum(x,dim=1)
        x  = self.net(x)      
        return x

class OurMSwebData(object):
  def __init__(self, av, elements, corpus, queries, sent_model):
    """
    elements: dict with keys = ["id", "text", "url"]
    corpus: list with each element being a set of element_ids
    query_ids: sub-list of corpus_ids
    """
    self.av = av
    self.device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    self.elements = elements
    self.corpus = corpus
    self.corpus_size = len(corpus)
    self.queries = queries

    self.corpus_tensors, self.corpus_set_sizes, self.corpus_mask_tensors = self.convert_items_to_tensors(self.corpus, sent_model)
    self.query_tensors, self.query_set_sizes, self.query_mask_tensors = self.convert_items_to_tensors(self.queries, sent_model)
    
    self.list_pos = []
    self.list_neg = []
    self.list_total_arranged_per_query = []
    self.labels_total_arranged_per_query = []

    logger.info("Initializing lpos and lneg")
    if av.DATASET_NAME == "MSWEB":
      subset_func = issubset_set
    else:
      subset_func = issubset_multiset

    for q in tqdm.tqdm(range(len(self.queries))):
      for i,c in enumerate(self.corpus):
        if subset_func(self.queries[q], c, av):
          self.list_pos.append(((q,i),1.0))
          self.list_total_arranged_per_query.append(((q,i),1.0))
          self.labels_total_arranged_per_query.append(1.0)
        else:
          self.list_neg.append(((q,i),0.0))
          self.list_total_arranged_per_query.append(((q,i),0.0))
          self.labels_total_arranged_per_query.append(0.0)

    logger.info(f"len(self.list_pos) = {len(self.list_pos)}")
    logger.info(f"len(self.list_neg) = {len(self.list_neg)}")
    logger.info(f"len(self.list_total_arranged_per_query) = {len(self.list_total_arranged_per_query)}, self.corpus_size = {self.corpus_size}, self.num_queries = {len(self.queries)}")

    self.eval_batches = {"map": dict(), "ap": dict()}

  def convert_items_to_tensors(self, items, sent_model):
    attid2sent = {e['id']: e['text'] for e in self.elements}
    sents = [e['text'] for e in self.elements]
    sent2id = {sent: i for (i,sent) in enumerate(sents)}
  
    #s_embeds = torch.tensor(sent_model.encode(sents)) 
    if self.av.EmbedType == "OneHot":
        s_embeds = torch.eye(len(sents),device=self.device)
    elif self.av.EmbedType == "Bert768": 
        s_embeds = torch.tensor(sent_model.encode(sents),device=self.device)
    else:
        raise NotImplementedError()

    sembed_dim = s_embeds[0].shape[0]
    logger.info(f"sembed_dim={sembed_dim}")
    
    nmax = max([len(c) for c in items])
    logger.info(f"nmax={nmax}")
    
    item_tensor = torch.zeros(len(items),nmax,sembed_dim,device=self.device)
    logger.info(f"item_tensor.shape = {item_tensor.shape}")
    
    set_sizes = []
    for i,c in enumerate(items):
        set_sizes.append(len(c))
        j = 0
        for attid in c:
            item_tensor[i,j,:] = s_embeds[sent2id[attid2sent[attid]]]
            j += 1
        for k in range(j,nmax):
            item_tensor[i,j,:] = torch.zeros(sembed_dim,device=self.device)
    
    # Generate mask
    mask_ids = torch.arange(item_tensor.size(1)).unsqueeze(0) < torch.FloatTensor(set_sizes).unsqueeze(-1)
    dim_after_inner_net_pass = self.av.IN_HIDDEN_LAYERS[-1] if self.av.DEEPSET_VER == 'UA' and self.av.IN_HIDDEN_LAYERS != [] else sembed_dim
    shape_after_inner_net_pass = (item_tensor.shape[0], nmax, dim_after_inner_net_pass)
    mask_tensor = torch.ones(shape_after_inner_net_pass, device=self.device)
    mask_tensor[~mask_ids] = 0

    return item_tensor, np.array(set_sizes), mask_tensor
    
  def create_stratified_batches(self):
    """
      Creates shuffled batches while maintaining class ratio
    """
    lpos = self.list_pos
    lneg = self.list_neg
    random.shuffle(lpos)
    random.shuffle(lneg)
    p2n_ratio = len(lpos)/len(lneg)
    npos = math.ceil((p2n_ratio/(1+p2n_ratio))*self.av.BATCH_SIZE)
    nneg = self.av.BATCH_SIZE-npos
    batches_pos, batches_neg = [],[]
    for i in range(0, len(lpos), npos):
      batches_pos.append(lpos[i:i+npos])
    for i in range(0, len(lneg), nneg):
      batches_neg.append(lneg[i:i+nneg])
     
    self.num_batches = min(len(batches_pos),len(batches_neg))  
    self.batches = [a+b for (a,b) in zip(batches_pos[:self.num_batches],batches_neg[:self.num_batches])]
    # self.alists = []
    # self.blists = []
    # self.scores = []
    # for b in tqdm.tqdm(self.batches):
    #   a,b = zip(*b)
    #   # print(list(a), list(b))
    #   g_pair = list(a)
    #   score = list(b)
      
    #   a,b = zip(*g_pair)
    #   # print(list(a), list(b))
    #   alist = list(a)
    #   blist = list(b)
    #   self.alists.append(alist)
    #   self.blists.append(blist)
    #   self.scores.append(score)
    return self.num_batches

  def create_batches_with_p2n(self):
    """
      Creates shuffled batches while maintaining given ratio
    """
    lpos = self.list_pos
    lneg = self.list_neg
    
    random.shuffle(lpos)
    random.shuffle(lneg)

    lpos_pair, lposs = zip(*lpos)
    lposa, lposb = zip(*lpos_pair)

    lneg_pair, lnegs = zip(*lneg)
    lnega, lnegb = zip(*lneg_pair)

    p2n_ratio = self.av.P2N
    batches_pos, batches_neg = [],[]
    as_pos, as_neg, bs_pos, bs_neg, ss_pos, ss_neg = [], [], [], [], [], []
    
    if av.BATCH_SIZE > 0:
      npos = math.ceil((p2n_ratio/(1+p2n_ratio))*self.av.BATCH_SIZE)
      nneg = self.av.BATCH_SIZE-npos
      self.num_batches = int(math.ceil(max(len(self.list_neg) / nneg, len(self.list_pos) / npos)))
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
      batches_pos.append(self.list_pos)
      batches_neg.append(self.list_neg)
     
    self.batches = [a+b for (a,b) in zip(batches_pos[:self.num_batches],batches_neg[:self.num_batches])]
    self.alists = [list(a+b) for (a,b) in zip(as_pos[:self.num_batches],as_neg[:self.num_batches])]
    self.blists = [list(a+b) for (a,b) in zip(bs_pos[:self.num_batches],bs_neg[:self.num_batches])]
    self.scores = [list(a+b) for (a,b) in zip(ss_pos[:self.num_batches],ss_neg[:self.num_batches])]
    self.alists_tensorized = [torch.tensor(list(a+b), device=self.device) for (a,b) in zip(as_pos[:self.num_batches],as_neg[:self.num_batches])]

    return self.num_batches

  def create_batches(self,list_all,metric,VAL_BATCH_SIZE=10000):
    """
      create batches as is and return number of batches created
    """

    if not self.eval_batches[metric]:
      self.batches = []
      self.alists = []
      self.blists = []
      self.scores = []
      self.alists_tensorized = []

      pair_all, score_all = zip(* list_all)
      as_all, bs_all = zip(* pair_all)
      for i in range(0, len(list_all), VAL_BATCH_SIZE):
        self.batches.append(list_all[i:i+VAL_BATCH_SIZE])
        self.alists.append(list(as_all[i:i+VAL_BATCH_SIZE]))
        self.blists.append(list(bs_all[i:i+VAL_BATCH_SIZE]))
        self.scores.append(list(score_all[i:i+VAL_BATCH_SIZE]))
        self.alists_tensorized.append(torch.tensor(list(as_all[i:i+VAL_BATCH_SIZE]), device=self.device))

      self.eval_batches[metric]['batches'] = self.batches
      self.eval_batches[metric]['alists'] = self.alists
      self.eval_batches[metric]['blists'] = self.blists
      self.eval_batches[metric]['scores'] = self.scores
      self.eval_batches[metric]['alists_tensorized'] = self.alists_tensorized
    else:
      self.batches = self.eval_batches[metric]['batches']
      self.alists = self.eval_batches[metric]['alists']
      self.blists = self.eval_batches[metric]['blists']
      self.scores = self.eval_batches[metric]['scores']
      self.alists_tensorized = self.eval_batches[metric]['alists_tensorized']
      
    self.num_batches = len(self.batches)  

    return self.num_batches
  
  def fetch_batched_data_by_id(self,i):
    """           
    """
    assert(i < self.num_batches)  
    batch = self.batches[i]
    
    a,b = zip(*batch)
    # print(list(a), list(b))
    g_pair = list(a)
    score = list(b)
    
    a,b = zip(*g_pair)
    # print(list(a), list(b))
    query_tensors = self.query_tensors[list(a)]
    query_mask_tensors = self.query_mask_tensors[list(a)]
    # query_set_sizes = [self.query_set_sizes[i] for i in a]
    corpus_tensors = self.corpus_tensors[list(b)]
    corpus_mask_tensors = self.corpus_mask_tensors[list(b)]
    # corpus_set_sizes = [self.corpus_set_sizes[i] for i in b]
    
    target = torch.tensor(score)
    return corpus_tensors, corpus_mask_tensors, query_tensors, query_mask_tensors, target, torch.tensor(list(a), device=self.device)

  def fetch_batched_data_by_id_optimized(self,i):
    """             
    """
    assert(i < self.num_batches)  
    # batch = self.batches[i]
    start_id_data = time.time()
    alist = self.alists[i]
    blist = self.blists[i]
    score = self.scores[i]
    # print(alist[:5], blist[:5], score[:5])
    id_data_time = time.time() - start_id_data
    
    start_load_tensors = time.time()
    query_tensors = self.query_tensors[alist]
    query_mask_tensors = self.query_mask_tensors[alist]
    # query_set_sizes = self.query_set_sizes[alist]
    load_tensors_time = time.time() - start_load_tensors

    start_set_size = time.time()
    corpus_tensors = self.corpus_tensors[blist]
    corpus_mask_tensors = self.corpus_mask_tensors[blist]
    # corpus_set_sizes = self.corpus_set_sizes[blist]
    set_size_time = time.time() - start_set_size

    start_score = time.time()
    target = torch.tensor(score)
    score_time = time.time() - start_score
    # print(f"id data: {id_data_time}, tensors: {load_tensors_time}, set size: {set_size_time}, score: {score_time}")
    return corpus_tensors, corpus_mask_tensors, query_tensors, query_mask_tensors, target, self.alists_tensorized[i]

def pairwise_ranking_loss_similarity(predPos, predNeg, av):
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = av.MARGIN + expanded_2 - expanded_1
    loss_fct = {'relu': torch.nn.ReLU(), 'softplus': torch.nn.Softplus()}
    loss = loss_fct[av.LOSS](ell)
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
    expanded_qid_2 = qidPos.unsqueeze(0).expand(n_1, n_2, dim)

    ell = av.MARGIN + expanded_2 - expanded_1
    loss_fct = {'relu': torch.nn.ReLU(), 'softplus': torch.nn.Softplus()}
    loss = loss_fct[av.LOSS](ell) * (expanded_qid_1 == expanded_qid_2)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(torch.sum(expanded_qid_1 == expanded_qid_2)), torch.sum(expanded_qid_1 == expanded_qid_2) / (n_1 * n_2)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                        type=str,   default="logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",                      type=bool,  default=True)
  ap.add_argument("--CUDA",                           type=int,  default=0)
  ap.add_argument("--RUN_TILL_ES",                    type=bool,  default=True)
  ap.add_argument("--has_cuda",                       type=bool,  default=torch.cuda.is_available())
  ap.add_argument("--ES",                             type=int,   default=5)
  ap.add_argument("--SKEW",                           type=int,   default=0)
  ap.add_argument("--NUM_RUNS",                       type=int,   default=10)
  ap.add_argument("--MAX_QUERIES",                    type=int,   default=100)
  ap.add_argument("--NUM_QUERIES",                    type=int,   default=500)
  ap.add_argument("--MARGIN",                         type=float, default=0.1)
  ap.add_argument("--WEIGHT_DECAY",                   type=float, default=5*10**-4)
  # ap.add_argument("--VAL_FRAC",                       type=float, default=0.2)
  # ap.add_argument("--TEST_FRAC",                       type=float, default=0.2)
  ap.add_argument("--IN_D",                           type=int,   default=50)
  ap.add_argument("--IN_HIDDEN_LAYERS",               type=int, nargs='*',  default=[10])
  ap.add_argument("--OUT_HIDDEN_LAYERS",              type=int, nargs='*',  default=[])
  ap.add_argument("--IN_RELU",                        action='store_true')
  ap.add_argument("--OUT_RELU",                       action='store_true')
  ap.add_argument("--PER_QUERY",                      action='store_true')
  ap.add_argument("--LAYER_NORMALIZE",                action='store_true')
  ap.add_argument("--HIDDEN_LAYERS",                  type=int, nargs='*',  default=[10])
  ap.add_argument("--BATCH_SIZE",                     type=int,   default=1024)
  ap.add_argument("--WRITE_EVERY",                     type=int,   default=500)
  ap.add_argument("--P2N",                            type=float,   default=1.0)
  ap.add_argument("--LEARNING_RATE",                  type=float, default=1e-5)
  ap.add_argument("--DIR_PATH",                       type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",                   type=str,   default="MSWEB", help="MSWEB/MSNBC")
  ap.add_argument("--DEEPSET_VER",                    type=str,   default="Default", help="Default/DefaultWithReLU/Identity/LinWoBias/LRLWoBias/LRLRLWoBias")
  ap.add_argument("--EmbedType",                      type=str,   default="OneHot", help="OneHot/Bert768")
  ap.add_argument("--TASK",                           type=str,   default="SymScore",help="SymScore/AsymScore/DotProduct")
  ap.add_argument("--CsubQ",                          action='store_true')
  ap.add_argument("--USE_SIG",                          action='store_true')
  ap.add_argument("--DESC",           type=str,   default="", help="")
  ap.add_argument("--LOSS",                           type=str,   default="relu", choices=["relu", "softplus"], help="SymScore/AsymScore")
  ap.add_argument("--TEST",                           action='store_true')

  av = ap.parse_args()
  
  IN_ARCH = "".join([f"L({dim})R" for dim in av.IN_HIDDEN_LAYERS])
  # print(IN_ARCH)
  IN_ARCH = IN_ARCH[:-1] if not av.IN_RELU else IN_ARCH
  # print(not av.IN_RELU, IN_ARCH)
  OUT_ARCH = "".join([f"L({dim})R" for dim in av.OUT_HIDDEN_LAYERS])
  OUT_ARCH = OUT_ARCH[:-1] if not av.OUT_RELU else OUT_ARCH
  av.TASK = av.TASK + f"{'csubq_' if av.CsubQ else ''}" + av.EmbedType + av.DEEPSET_VER  + (f"_LAYERS{':'.join([str(k) for k in av.HIDDEN_LAYERS])}" if av.DEEPSET_VER.startswith("Default") \
                    else f"{IN_ARCH}sum{OUT_ARCH}_LAYERNORMALIZE{av.LAYER_NORMALIZE}" if av.DEEPSET_VER == "UA" \
                    else "")
  skew = "skew"+str(av.SKEW) if av.SKEW!=0 else ""
  sig = "sig" if av.USE_SIG else ""
  av.TASK = av.TASK + "_" + av.DATASET_NAME +skew +sig+ "_in_d" + str(av.IN_D) + "_batchSz" + str(av.BATCH_SIZE) + \
    "_ES" + str(av.ES) + f"_P2N{av.P2N}_LEARNING_RATE{av.LEARNING_RATE}_NUM_QUERIES{av.NUM_QUERIES}_MAX_QUERIES{av.MAX_QUERIES}_MARGIN{av.MARGIN}_LOSS{av.LOSS}_PER_QUERY{av.PER_QUERY}"
  av.logpath = av.logpath+"_"+av.TASK+datetime.now().isoformat()
  set_log(av)
  logger.info("Command line")
  logger.info('\n'.join(sys.argv[:]))

  if av.want_cuda and av.has_cuda:
    torch.cuda.set_device(av.CUDA)
    logger.info(f"using cuda: {torch.cuda.current_device()}")
    
  # Set random seeds
  seed = 4
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)
  torch.backends.cudnn.deterministic = False
#  torch.backends.cudnn.benchmark = True
  # torch.autograd.set_detect_anomaly(True)

  if av.TEST:
    test(av)
  else:
    train(av)



#CUDA_VISIBLE_DEVICES=2 python3 -m src.msweb_data_generator_and_trainer --LEARNING_RATE 1e-3 --TASK AsymScore --MARGIN 1.0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LOSS relu --DEEPSET_VER Identity --EmbedType OneHot 
#CUDA_VISIBLE_DEVICES=2 python3 -m src.msweb_data_generator_and_trainer --LEARNING_RATE 1e-3 --TASK AsymScore --MARGIN 1.0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LOSS relu --DEEPSET_VER LinWoBias --EmbedType OneHot 
#CUDA_VISIBLE_DEVICES=3 python3 -m src.msweb_data_generator_and_trainer --LEARNING_RATE 1e-3 --TASK AsymScore --MARGIN 1.0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LOSS relu --DEEPSET_VER LRLWoBias --EmbedType OneHot 
#CUDA_VISIBLE_DEVICES=3 python3 -m src.msweb_data_generator_and_trainer --LEARNING_RATE 1e-3 --TASK AsymScore --MARGIN 1.0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LOSS relu --DEEPSET_VER LRLRLWoBias --EmbedType OneHot 

#CUDA_VISIBLE_DEVICES=2 python3 -m src.msweb_data_generator_and_trainer --LEARNING_RATE 1e-3 --TASK AsymScore --MARGIN 1.0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LOSS relu --DEEPSET_VER Identity --EmbedType Bert768 
#CUDA_VISIBLE_DEVICES=2 python3 -m src.msweb_data_generator_and_trainer --LEARNING_RATE 1e-3 --TASK AsymScore --MARGIN 1.0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LOSS relu --DEEPSET_VER LinWoBias --EmbedType Bert768 
#CUDA_VISIBLE_DEVICES=0 python3 -m src.msweb_data_generator_and_trainer --LEARNING_RATE 1e-3 --TASK AsymScore --MARGIN 1.0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LOSS relu --DEEPSET_VER LRLWoBias --EmbedType Bert768 
#CUDA_VISIBLE_DEVICES=1 python3 -m src.msweb_data_generator_and_trainer --LEARNING_RATE 1e-3 --TASK AsymScore --MARGIN 1.0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LOSS relu --DEEPSET_VER LRLRLWoBias --EmbedType Bert768 

#RA's comms
#python -m src.msweb_data_generator_and_trainer --TASK SymScore --MARGIN 0.1 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --IN_RELU 
#python -m src.msweb_data_generator_and_trainer --TASK AsymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --IN_RELU 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK AsymScore --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType OneHot --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 10 --OUT_RELU --PLOT_EVERY 2000 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType OneHot --IN_HIDDEN_LAYERS 500 10 --OUT_HIDDEN_LAYERS --PLOT_EVERY 2000 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType OneHot --IN_HIDDEN_LAYERS 500 10 --OUT_HIDDEN_LAYERS --WRITE_EVERY 2000


#IR's
#python -m src.msweb_data_generator_and_trainer --TASK SigmoidAsymScore --MARGIN 0.1 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --IN_RELU 
#python -m src.msweb_data_generator_and_trainer --TASK SigmoidSymScore --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --IN_RELU 

#python -m src.msweb_data_generator_and_trainer --TASK DotProduct --MARGIN 0.1 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --IN_RELU  
#python -m src.msweb_data_generator_and_trainer --TASK DotProduct --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --IN_RELU  

#python -m src.msweb_data_generator_and_trainer --TASK WeightedJaccard --MARGIN 0.1 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --IN_RELU  
#python -m src.msweb_data_generator_and_trainer --TASK WeightedJaccard --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --IN_RELU  



#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK AsymScore --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK AsymScore --MARGIN 1.0 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK AsymScore --MARGIN 0.1 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK AsymScore --MARGIN 0.1 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 0.1 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 0.1 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.1 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.1 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.5 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  



#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 

#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK AsymScore --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK AsymScore --MARGIN 1.0 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK AsymScore --MARGIN 0.1 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1   




#===================
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=4  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=5  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=6  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=7  

#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  --SKEW=3
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  --SKEW=3

#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.1 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.1 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3 


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=4
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  --SKEW=4
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  --SKEW=4 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  --SKEW=4


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=4 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=4 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.1 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=4 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.1 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=4 


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=5
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  --SKEW=5
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  --SKEW=5 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 0.1 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  --SKEW=5


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=5 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=5 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.1 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=5 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 0.1 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=5 



#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 0.1 --CUDA 0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 0.1 --CUDA 6 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=4 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 0.1 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=5 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 0.1 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=6 

#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 1.0 --CUDA 7 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 1.0 --CUDA 5 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=4 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 1.0 --CUDA 0 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=5 
#=============TODO============
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 1.0 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 0.1 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=3 
#####################################################################333


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1  
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK SigmoidAsymScore --MARGIN 1.0 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1  

#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK SymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK SymScore --MARGIN 1.0 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  --SKEW=1
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK SymScore --MARGIN 0.1 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU  --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK SymScore --MARGIN 0.1 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU  --SKEW=1

#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK DotProduct --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK DotProduct --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK DotProduct --MARGIN 0.1 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK DotProduct --MARGIN 0.1 --CUDA 4 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 


#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK WeightedJaccard --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK WeightedJaccard --MARGIN 1.0 --CUDA 3 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK WeightedJaccard --MARGIN 0.1 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS --OUT_HIDDEN_LAYERS 294 --OUT_RELU --SKEW=1 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK WeightedJaccard --MARGIN 0.1 --CUDA 1 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 


#################################
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK SymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 --USE_SIG 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK DotProduct --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 --USE_SIG 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK WeightedJaccard --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=1 --USE_SIG 

#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK SymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --USE_SIG 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK DotProduct --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --USE_SIG 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSWEB --TASK WeightedJaccard --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --USE_SIG 

 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3 --USE_SIG 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3 --USE_SIG 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=3 --USE_SIG 

#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK SymScore --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=5 --USE_SIG 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK DotProduct --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=5 --USE_SIG 
#python -m src.msweb_data_generator_and_trainer --DATASET_NAME MSNBC --TASK WeightedJaccard --MARGIN 1.0 --CUDA 2 --MAX_QUERIES 100 --NUM_QUERIES 500 --LEARNING_RATE 1e-3 --LOSS relu --BATCH_SIZE 1024 --P2N 1.0 --DEEPSET_VER UA --ES 50 --EmbedType Bert768 --IN_HIDDEN_LAYERS 294 --OUT_HIDDEN_LAYERS  --IN_RELU --SKEW=5 --USE_SIG 




