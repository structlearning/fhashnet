import numpy as np
import random
import time
import torch
from src.sample import generate_samples
import os
import pickle
from common import logger, set_log
from src.utils import cudavar
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import psutil
from src.earlystopping import EarlyStoppingModule

#from src.fmap_trainer import *
from src.fmap_trainer import FmapTrainer,AsymFmapTrainer
import src.hashcode_trainer1 as htr
import src.hashcode_trainer_hingeemb as htr_hingeemb
import src.endtoend_hashcode_trainer1 as e2ehtr
import src.endtoend_hashcode_trainer2 as e2e2htr
import src.endtoend_hashcode_trainer3 as e2e3htr
import src.endtoend_hashcode_trainer4 as e2e4htr
from collections import defaultdict
from drhash import WeightedMinHash
from scipy.sparse import csc_matrix


def get_wmh_hcode(av, embeds):
    wmh = WeightedMinHash.WeightedMinHash(csc_matrix(embeds.T), av.hcode_dim)
    if av.WMH =="minhash":   
        wmh_op = wmh.minhash()
    elif av.WMH =="gollapudi2":   
        wmh_op = wmh.gollapudi2() 
    elif av.WMH =="icws":   
        wmh_op = wmh.icws()  
    elif av.WMH =="licws":   
        wmh_op = wmh.licws()  
    elif av.WMH =="pcws":   
        wmh_op = wmh.pcws()  
    elif av.WMH =="ccws":   
        wmh_op = wmh.ccws()  
    elif av.WMH =="i2cws":   
        wmh_op = wmh.i2cws()  
    elif av.WMH =="chum":   
        wmh_op = wmh.chum()  
    elif av.WMH =="shrivastava":   
        wmh_op = wmh.shrivastava()  
    else:
        raise NotImplementedError()
        
    if len(wmh_op) ==3 : 
        hcodes = np.array([str(x)+str(y) for x,y  \
                  in zip(wmh_op[0].flatten(),wmh_op[1].flatten())]).reshape(wmh_op[0].shape)
    elif len(wmh_op) ==2 : 
        hcodes = np.array([str(x) for x in wmh_op[0].flatten()]).reshape(wmh_op[0].shape)
    else:
        raise NotImplementedError()

    return hcodes.squeeze()
        

def check_pretrained_fmaps(av):
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    #if av.trained_cosine_fmap_pickle_fp !="":
    HID_ARCH = "".join([f"RL_{dim}_" for dim in av.FMAP_HIDDEN_LAYERS])
    temp_IN_ARCH = "L" +  HID_ARCH 
    temp_c_sub = "CsubQ" if av.CsubQ else ""

    curr_DESC = av.DESC
    temp_DESC= av.TASK+temp_c_sub+"_"+av.DATASET_NAME+ "_MARGIN" + str(av.MARGIN) +"_muse" + str(av.m_use) + "_T" + str(av.T) + "_Scale" + str(av.SCALE) + "_TrFmapDim" + str(av.tr_fmap_dim)+ "_fmapSelect_" + str(av.FMAP_SELECT)+ "_SpecialFhash_" + (av.SPECIAL_FHASH if av.SPECIAL_FHASH!="" else "Asym")+  "_LOSS_" + av.FMAP_LOSS_TYPE +  "_arch_"+ temp_IN_ARCH  + ("_fmapBCE" if av.USE_FMAP_BCE else "") + ("_fmapBCE2" if av.USE_FMAP_BCE2 else "")+ ("_fmapBCE3" if av.USE_FMAP_BCE3 else "") + ("_fmapMSE" if av.USE_FMAP_MSE else "")+ ("_fmapPQR" if av.USE_FMAP_PQR else "")
    av.DESC = temp_DESC
    #checking existence of best val model 
    bvalmodel_pathname = av.DIR_PATH + "/bestValidationModels/"+temp_DESC#+av.DATASET_NAME+"_"+temp_DESC
    #assert( os.path.exists(bvalmodel_pathname )), print(bvalmodel_pathname)

    #loading bestvalmodel
    es = EarlyStoppingModule(av,av.ES)
    
    checkpoint = es.load_best_model(device='cpu')
    av.DESC = curr_DESC
    if av.FMAP_LOSS_TYPE == "AsymFmapCos": 
        model = AsymFmapTrainer(av).to(device)
    elif av.FMAP_LOSS_TYPE == "FmapCos": 
        model = FmapTrainer(av).to(device)
    else:
        raise NotImplementedError()

    model.load_state_dict(checkpoint['model_state_dict'])    
    model_weights_np = {}
    if av.FMAP_LOSS_TYPE == "AsymFmapCos": 
        model_weights_np['np_w_q'] = model.init_net[0].weight.cpu().detach().numpy()
        model_weights_np['np_b_q'] = model.init_net[0].bias.cpu().detach().numpy()
        model_weights_np['np_w_c'] = model.init_cnet[0].weight.cpu().detach().numpy()
        model_weights_np['np_b_c'] = model.init_cnet[0].bias.cpu().detach().numpy()
    elif av.FMAP_LOSS_TYPE == "FmapCos": 
        model_weights_np['np_w'] = model.init_net[0].weight.cpu().detach().numpy()
        model_weights_np['np_b'] = model.init_net[0].bias.cpu().detach().numpy()
    else:
        raise NotImplementedError()

    #checking existence of dumped trained fmaps
    pathname = av.DIR_PATH + "/data/fmapPickles/"+temp_DESC+"_fmap_mat.pkl"
    print(pathname)

    assert( os.path.exists(pathname )), print(pathname)
    #tr_fmap_data = pickle.load(open(pathname, "rb"))
    #return tr_fmap_data,model_weights_np
    return None,model_weights_np


def check_pretrained_hashcodes(av):
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    #if av.trained_cosine_fmap_pickle_fp !="":
    HID_ARCH = "".join([f"RL_{dim}_" for dim in av.HIDDEN_LAYERS])
    FMAP_HID_ARCH = "".join([f"RL_{dim}_" for dim in av.FMAP_HIDDEN_LAYERS])
    FMAP_IN_ARCH = "L" +  FMAP_HID_ARCH
    temp_IN_ARCH = "L" +  HID_ARCH + ("Tanh" if not av.NO_TANH else "") + ("Lnorm" if av.LAYER_NORMALIZE else "")+\
          ("InitGauss" if av.INIT_GAUSS else "Init_KH")
    temp_c_sub = "CsubQ" if av.CsubQ else ""
    
    curr_DESC = av.DESC

    sc_loss_subset = str(av.sc_subset_size) if (av.LOSS_TYPE=="sc_loss" and av.sc_subset_size!=8) else ""
    
    temp_DESC= av.TASK+temp_c_sub+"_"+av.DATASET_NAME+ "_MARGIN" + str(av.MARGIN) +"_muse" + str(av.m_use) +\
    "_T" + str(av.T) + "_Scale" + str(av.SCALE) + "_hcode" + str(av.hcode_dim)+ "_fmapSelect_" +\
    str(av.FMAP_SELECT)+ "_SpecialFhash_" + (av.SPECIAL_FHASH if av.SPECIAL_FHASH!="" else "Asym")+\
    "_LOSS_" + av.LOSS_TYPE + sc_loss_subset + (("_tanh_temp"+str(av.TANH_TEMP)) if not av.NO_TANH  else "")+\
    (("_fence_" + str(av.FENCE_LAMBDA)) if av.FENCE_LAMBDA!=0.0 else "")+\
    (("_decorr_" + str(av.DECORR_LAMBDA)) if av.DECORR_LAMBDA!=0.0 else "")+\
    (("_C1loss_" + str(av.C1_LAMBDA)) if av.C1_LAMBDA!=0.0 else "") +\
    (("_SClossMargin" + str(av.SCLOSS_MARGIN)) if av.SCLOSS_MARGIN!=1.0 else "") +\
    (("_weaksup_" + str(av.WEAKSUP_LAMBDA)) if av.WEAKSUP_LAMBDA!=0.0 else "")+\
    "_arch_"+ temp_IN_ARCH  +("SignEval" if av.SIGN_EVAL else "NoSignEval") + \
    ("_pretrained_fmap"+  "_FMAPLOSS_" + av.FMAP_LOSS_TYPE +  "_fmapArch_"+ FMAP_IN_ARCH + "_TrFmapDim" + str(av.tr_fmap_dim))+\
    ("_fmapBCE" if av.USE_FMAP_BCE else "") + ("_fmapBCE2" if av.USE_FMAP_BCE2 else "")+ ("_fmapBCE3" if av.USE_FMAP_BCE3 else "") + ("_fmapMSE" if av.USE_FMAP_MSE else "") + ("_fmapPQR" if av.USE_FMAP_PQR else "")



    av.DESC = temp_DESC
    #checking existence of best val model 
    bvalmodel_pathname = av.DIR_PATH + "/bestValidationModels/"+temp_DESC#+av.DATASET_NAME+"_"+temp_DESC
    #assert( os.path.exists(bvalmodel_pathname )), print(bvalmodel_pathname)

    #loading bestvalmodel
    es = EarlyStoppingModule(av,av.ES)

    checkpoint = es.load_best_model(device='cpu')
    av.DESC = curr_DESC
    if av.LOSS_TYPE == "sc_loss" or av.LOSS_TYPE == "permgnn_loss": 
        model = htr.HashCodeTrainer(av).to(device)
    else:
        raise NotImplementedError()

    model.load_state_dict(checkpoint['model_state_dict'])    
    model_weights_np = {}
    
    if av.LOSS_TYPE == "sc_loss" or  av.LOSS_TYPE == "permgnn_loss": 
        model_weights_np['num_layers'] = int((len(model.init_net)+1)/2)
        for idx in range(model_weights_np['num_layers']):
            model_weights_np[idx] = {}
            model_weights_np[idx]['np_w'] = model.init_net[2*idx].weight.cpu().detach().numpy()
            model_weights_np[idx]['np_b'] = model.init_net[2*idx].bias.cpu().detach().numpy()
        #model_weights_np['np_w'] = model.init_net[0].weight.cpu().detach().numpy()
        #model_weights_np['np_b'] = model.init_net[0].bias.cpu().detach().numpy()
    else:
        raise NotImplementedError()

    #checking existence of dumped trained hashcodes
    pathname = av.DIR_PATH + "/data/hashcodePickles/"+temp_DESC+"_hashcode_mat.pkl"
    pathname1 = av.DIR_PATH + "/data/hashcodePickles/"+temp_DESC+"_hashcode_mat"
    #print(pathname)

    assert( os.path.exists(pathname1 ) or os.path.exists(pathname )), print(pathname)
    #tr_fmap_data = pickle.load(open(pathname, "rb"))
    #return tr_fmap_data,model_weights_np
    return None,model_weights_np


def check_pretrained_hashcodes_hingeemb(av):
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    HID_ARCH = "".join([f"RL_{dim}_" for dim in av.HIDDEN_LAYERS])
    temp_IN_ARCH = "L" +  HID_ARCH + ("Tanh" if not av.NO_TANH else "") + ("Lnorm" if av.LAYER_NORMALIZE else "")+\
          ("InitGauss" if av.INIT_GAUSS else "Init_KH")
    temp_c_sub = "CsubQ" if av.CsubQ else ""
    silver = "SilverSig" if av.SilverSig else ""
    dothash = "DotHash" if av.HASH_MODE=="dot" else ""
    dot2hash = "Dot2Hash" if av.HASH_MODE=="dot2" else ""
    flora = "Flora" if av.HASH_MODE=="flora" else ""
    sc_loss_subset = str(av.sc_subset_size) if (av.LOSS_TYPE=="sc_loss_hingeemb" and av.sc_subset_size!=8) else ""
    
    curr_DESC = av.DESC
  
    temp_DESC= av.TASK+dothash+dot2hash+silver+flora+temp_c_sub+"_"+av.DATASET_NAME+ "_MARGIN" + str(av.MARGIN) +\
    "_muse" + str(av.m_use) + "_T" + str(av.T) + "_Scale" + str(av.SCALE) +\
    "_hcode" + str(av.hcode_dim)+ "_fmapSelect_" + str(av.FMAP_SELECT)+\
    "_SpecialFhash_" + (av.SPECIAL_FHASH if av.SPECIAL_FHASH!="" else "Asym")+\
    "_LOSS_" + av.LOSS_TYPE + sc_loss_subset + (("_tanh_temp"+str(av.TANH_TEMP)) if not av.NO_TANH  else "")+\
    (("_fence_" + str(av.FENCE_LAMBDA)) if av.FENCE_LAMBDA!=0.0 else "")+\
    (("_decorr_" + str(av.DECORR_LAMBDA)) if av.DECORR_LAMBDA!=0.0 else "")+\
    (("_C1loss_" + str(av.C1_LAMBDA)) if av.C1_LAMBDA!=0.0 else "") +\
    (("_SClossMargin" + str(av.SCLOSS_MARGIN)) if av.SCLOSS_MARGIN!=1.0 else "") +\
    (("_weaksup_" + str(av.WEAKSUP_LAMBDA)) if av.WEAKSUP_LAMBDA!=0.0 else "")\
    +"_arch_"+ temp_IN_ARCH  +("SignEval" if av.SIGN_EVAL else "NoSignEval") 

    av.DESC = temp_DESC

    
    #checking existence of best val model 
    bvalmodel_pathname = av.DIR_PATH + "/bestValidationModels/"+temp_DESC#+av.DATASET_NAME+"_"+temp_DESC
    #assert( os.path.exists(bvalmodel_pathname )), print(bvalmodel_pathname)

    #loading bestvalmodel
    es = EarlyStoppingModule(av,av.ES)

    checkpoint = es.load_best_model(device='cpu')
    av.DESC = curr_DESC
    if av.LOSS_TYPE == "sc_loss_hingeemb" or  av.LOSS_TYPE == "permgnn_loss_hingeemb" or av.LOSS_TYPE == "flora_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb2":
        model = htr_hingeemb.HashCodeTrainer(av).to(device)
    else:
        raise NotImplementedError()

    model.load_state_dict(checkpoint['model_state_dict'])    
    model_weights_np = {}
    if av.LOSS_TYPE == "sc_loss_hingeemb" or av.LOSS_TYPE == "permgnn_loss_hingeemb" or av.LOSS_TYPE == "flora_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb2": 
        model_weights_np['num_layers'] = int((len(model.init_net)+1)/2)
        for idx in range(model_weights_np['num_layers']):
            model_weights_np[idx] = {}
            model_weights_np[idx]['np_w'] = model.init_net[2*idx].weight.cpu().detach().numpy()
            model_weights_np[idx]['np_b'] = model.init_net[2*idx].bias.cpu().detach().numpy()
        #model_weights_np['np_w'] = model.init_net[0].weight.cpu().detach().numpy()
        #model_weights_np['np_b'] = model.init_net[0].bias.cpu().detach().numpy()
    else:
        raise NotImplementedError()

    if av.LOSS_TYPE == "flora_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb2" : 
        model_weights_np['np_w_q'] = model.init_qnet[0].weight.cpu().detach().numpy()
        model_weights_np['np_b_q'] = model.init_qnet[0].bias.cpu().detach().numpy()
        model_weights_np['np_w_c'] = model.init_cnet[0].weight.cpu().detach().numpy()
        model_weights_np['np_b_c'] = model.init_cnet[0].bias.cpu().detach().numpy()
        #model_weights_np['np_w'] = model.init_net[0].weight.cpu().detach().numpy()
        #model_weights_np['np_b'] = model.init_net[0].bias.cpu().detach().numpy()
    

    #checking existence of dumped trained hashcodes
    pathname = av.DIR_PATH + "/data/hashcodePickles/"+temp_DESC+"_hashcode_mat.pkl"
    pathname1 = av.DIR_PATH + "/data/hashcodePickles/"+temp_DESC+"_hashcode_mat"
    #print(pathname)

    assert( os.path.exists(pathname1 ) or os.path.exists(pathname )), print(pathname)
    #tr_fmap_data = pickle.load(open(pathname, "rb"))
    #return tr_fmap_data,model_weights_np
    return None,model_weights_np
    #else:
    #    return None

def check_pretrained_hashcodes_endtoend(av):
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    HID_ARCH = "".join([f"RL_{dim}_" for dim in av.HIDDEN_LAYERS])
    temp_IN_ARCH = "L" +  HID_ARCH + ("Tanh" if not av.NO_TANH else "") + ("Lnorm" if av.LAYER_NORMALIZE else "")+\
          ("InitGauss" if av.INIT_GAUSS else "Init_KH")
    temp_c_sub = "CsubQ" if av.CsubQ else ""
    
    curr_DESC = av.DESC
  
    temp_DESC= av.TASK+av.E2EVER+temp_c_sub+"_"+av.DATASET_NAME+ "_MARGIN" + str(av.MARGIN) +\
    "_muse" + str(av.m_use) + "_T" + str(av.T) + "_Scale" + str(av.SCALE) +\
    "_hcode" + str(av.hcode_dim)+ "_fmapSelect_" + str(av.FMAP_SELECT)+\
    "_SpecialFhash_" + (av.SPECIAL_FHASH if av.SPECIAL_FHASH!="" else "Asym")+\
    "_LOSS_" + av.LOSS_TYPE + (("_tanh_temp"+str(av.TANH_TEMP)) if not av.NO_TANH  else "")+\
    (("_fence_" + str(av.FENCE_LAMBDA)) if av.FENCE_LAMBDA!=0.0 else "")+\
    (("_decorr_" + str(av.DECORR_LAMBDA)) if av.DECORR_LAMBDA!=0.0 else "")+\
    (("_C1loss_" + str(av.C1_LAMBDA)) if av.C1_LAMBDA!=0.0 else "") +\
    (("_SClossMargin" + str(av.SCLOSS_MARGIN)) if av.SCLOSS_MARGIN!=1.0 else "") +\
    (("_weaksup_" + str(av.WEAKSUP_LAMBDA)) if av.WEAKSUP_LAMBDA!=0.0 else "")\
    +"_arch_"+ temp_IN_ARCH  +("SignEval" if av.SIGN_EVAL else "NoSignEval") 

    av.DESC = temp_DESC

    
    #checking existence of best val model 
    bvalmodel_pathname = av.DIR_PATH + "/bestValidationModels/"+temp_DESC#+av.DATASET_NAME+"_"+temp_DESC
    #assert( os.path.exists(bvalmodel_pathname )), print(bvalmodel_pathname)

    #loading bestvalmodel
    es = EarlyStoppingModule(av,av.ES)

    checkpoint = es.load_best_model()
    av.DESC = curr_DESC
    if av.LOSS_TYPE == "sc_loss":# or  av.LOSS_TYPE == "permgnn_loss_hingeemb":
        if av.E2EVER=="e2e3": 
            model = e2e3htr.HashCodeTrainer(av).to(device)
        elif av.E2EVER=="e2e4": 
            model = e2e4htr.HashCodeTrainer(av).to(device)
        elif av.E2EVER=="e2e2": 
            model = e2e2htr.HashCodeTrainer(av).to(device)
        else:
            model = e2ehtr.HashCodeTrainer(av).to(device)
    else:
        raise NotImplementedError()

    model.load_state_dict(checkpoint['model_state_dict'])    
    model_weights_np = {}
    if av.LOSS_TYPE == "sc_loss":# or av.LOSS_TYPE == "permgnn_loss_hingeemb":
        model_weights_np['num_layers'] = int((len(model.init_net)+1)/2)
        for idx in range(model_weights_np['num_layers']):
            model_weights_np[idx] = {}
            model_weights_np[idx]['np_w'] = model.init_net[2*idx].weight.cpu().detach().numpy()
            model_weights_np[idx]['np_b'] = model.init_net[2*idx].bias.cpu().detach().numpy()
    else:
        raise NotImplementedError()

    #checking existence of dumped trained hashcodes
    pathname = av.DIR_PATH + "/data/hashcodePickles/"+temp_DESC+"_hashcode_mat.pkl"
    pathname1 = av.DIR_PATH + "/data/hashcodePickles/"+temp_DESC+"_hashcode_mat"
    #print(pathname)

    assert( os.path.exists(pathname1 ) or os.path.exists(pathname )), print(pathname)
    tr_fmap_data = pickle.load(open(pathname, "rb"))
    return tr_fmap_data,model_weights_np
    #else:
    #    return None



def user_time():
  return psutil.Process().cpu_times().user  
# psutil.Process().cpu_times().user
# time.process_time()
# time.time()
def optimized_cosine_similarity(a, b1):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the cosine similarity cos(a[i],b[j]) for all i and j.
    :return: Matrix with res[i][j]  = \sum(a[i]*b[j])
    """
    eps = 1e-8
    return (a@b1.T)/((np.linalg.norm(a,axis=1)[:,None] + eps)*(np.linalg.norm(b1,axis=1)[None,:]+eps))

def np_sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_asym_sim(av, query_embeds, corpus_embeds):
    '''
        query_embeds: 1 x d or m x d
        corpus_embeds: m x d
        output sim: m x 1
    '''
    #sim =  np.sum(av.T - np.maximum(0, query_embeds-corpus_embeds), axis=1)
    sig_in = np.sum(np.maximum(0, query_embeds-corpus_embeds), axis=1)
    sig_out = np_sigmoid(av.sigmoid_a*sig_in + av.sigmoid_b)
    return sig_out

def dot_sim(a, b):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the dot similarity a[i]*b[j] for all i and j.
    :return: Matrix with res[i][j]  = \sum(a[i]*b[j])
    """
    return (a@b.T) # (a[:,None,:]*b[None,:,:]).sum(-1) # 

def asym_sim(av, query_embeds, corpus_embeds):
    '''
        query_embeds: 1 x d or m x d
        corpus_embeds: m x d
        output sim: m x 1
    '''
    #sim =  np.sum(av.T - np.maximum(0, query_embeds-corpus_embeds), axis=1)
    sim =  -np.sum(np.maximum(0, query_embeds-corpus_embeds), axis=1)
    return sim

def wjac_sim(a, b):
    '''
        query_embeds: 1 x d 
        corpus_embeds: m x d
        output sim: m x 1
    '''
    return np.minimum(a,b).sum(-1)/(np.maximum(a,b).sum(-1)+1e-8)

def fetch_samples(av):
    """
        Fetch 'm_load' samples of w given T, a, b
    """
    assert (av.m_load == av.m_use*av.embed_dim) , f"m_load: {av.m_load} should be equal to m_use*embed_dim: { av.m_use*av.embed_dim}"
    logger.info('Fetching samples ...')
    samples_fp = "./data/samples_num" + str(av.m_load) + "_a"+ str(av.a) +\
                        "_b" + str(av.b) + "_T" + str(av.T) + ".pkl"
    if os.path.exists(samples_fp):
        all_d = pickle.load(open(samples_fp,"rb"))
        logger.info(f"Fetching samples from  {samples_fp}")
    else:
        logger.info(f"Samples not found, so generating samples and dumping to {samples_fp}")
        all_d = generate_samples(av.m_load, av.T, av.a, av.b)
        pickle.dump(all_d, open(samples_fp, "wb"))
    logger.info('Samples fetched')
    return np.float32(all_d['samples']), np.float32(all_d['pdf'])

def fetch_gaussian_hyperplanes(av, fmap_dim):
    """
        Based on the dimention of the feature map
        Load/generate specified no of gaussian hyperplanes
        av.hcode_dim specifies no of planes to generate
    """
    logger.info('Fetching gaussian hplanes ...')
    fp = f'./data/gauss_hplanes_hdim{av.hcode_dim}_fmap_dim{fmap_dim}.pkl'
    if os.path.exists(fp):
        all_data = pickle.load(open(fp,'rb'))
        hplanes = all_data['hplanes']
    else:
        hplanes = np.random.normal(size=(fmap_dim, av.hcode_dim))
        all_data = {
            'hplanes': hplanes,
            'fmap_dim': fmap_dim,
            'hcode_dim': av.hcode_dim
        }
        pickle.dump(all_data, open(fp,"wb"))
    logger.info("From %s", fp)
    return hplanes


def generate_fourier_map_tensorized_np(av, embeds,ws,sqrt_pdfs, isQuery=False): 
    """
        Given some value of T, limit, a,b
        Fetch/generate prob samples
        compute map and return 
    """
    # type_ = "Query" if isQuery else "Corpus"

    R_G = 2 * (np.sin(ws * av.T/2))**2 / ws**2 + \
            av.T * np.sin(ws * av.T) / ws
    I_G = np.sin(ws * av.T) / ws**2 - \
            av.T * np.cos(ws * av.T) / ws
    #d = embeds.shape[1]
    sign_RG = np.sign(R_G)
    sign_IG = np.sign(I_G)
    sqrt_abs_RG = np.sqrt(np.abs(R_G))
    sqrt_abs_IG = np.sqrt(np.abs(I_G))

    embeds_rep = np.repeat(embeds,av.m_use,axis=-1)
  
    thetas = embeds_rep*ws

    cos_theta_by_sqrt_pdf = np.cos(thetas) / sqrt_pdfs
    sin_theta_by_sqrt_pdf = np.sin(thetas) / sqrt_pdfs
    if isQuery:
        fmap1 = sign_RG * sqrt_abs_RG * cos_theta_by_sqrt_pdf 
        fmap2 = sign_RG * sqrt_abs_RG * sin_theta_by_sqrt_pdf
        fmap3 = - sign_IG * sqrt_abs_IG * sin_theta_by_sqrt_pdf
        fmap4 = sign_IG * sqrt_abs_IG * cos_theta_by_sqrt_pdf
    else:
        fmap1 =  sqrt_abs_RG * cos_theta_by_sqrt_pdf
        fmap2 =  sqrt_abs_RG * sin_theta_by_sqrt_pdf
        fmap3 =  sqrt_abs_IG * cos_theta_by_sqrt_pdf
        fmap4 =  sqrt_abs_IG * sin_theta_by_sqrt_pdf
    
    fmaps = np.hstack([fmap1,fmap2,fmap3,fmap4])#.numpy()
    #fmaps = np.concatenate((fmap1,fmap2,fmap3,fmap4))#.numpy()
#     fmaps = np.hstack([fmap1.reshape(-1,10),\
#                       fmap2.reshape(-1,10),\
#                       fmap3.reshape(-1,10),\
#                       fmap4.reshape(-1,10)]).flatten()
  
    return fmaps



def generate_fourier_map(av, embeds, ws, pdfs, isQuery=False): 
    """
        Given some value of T, limit, a,b
        Fetch/generate prob samples
        compute map and return 
    """
    type_ = "Query" if isQuery else "Corpus"
    #logger.info(f'Fetching fourier map for {type_}')
    #fp = get_fourier_map_fname(av,type_)
    #
    #if os.path.exists(fp) and not isQuery and not(av.DEBUG):
    #    all_d = pickle.load(open(fp, "rb"))
    #    fmaps = all_d['fmaps']
    #else:
    #    logger.info(f'Fourier map not found, so generating fourier map for {type_}')
    #ws, pdfs = fetch_samples(av)
    assert len(ws) == len(pdfs)
    # logger.info(f"#ws={len(ws)}")
    #assert av.m_load == len(ws)
    d = embeds.shape[1]
    #NOTE(): I'd like to enforce equality below. Let's discuss. 
    assert av.m_use * d <= av.m_load, "We need more samples to generate fourier map"
    ws = ws[:av.m_use * d].reshape((d, av.m_use))
    sqrt_pdfs = np.sqrt(pdfs[:av.m_use * d].reshape((d, av.m_use)))
    fmap_single_dim = [None for _ in range(d)]
    for i in range(d):
        logger.debug(f'i={i}')
        ws_single_dim = torch.tensor(ws[i,:].reshape((1,av.m_use)))
        sqrt_pdfs_single_dim = torch.tensor(sqrt_pdfs[i,:].reshape((1,av.m_use)))
        logger.debug("hello -1")
        embeds_single_dim = torch.tensor(embeds[:,i].reshape((embeds.shape[0],1)))
        logger.debug("hello 0")
        R_G = 2 * (torch.sin(ws_single_dim * av.T/2))**2 / ws_single_dim**2 + \
                av.T * torch.sin(ws_single_dim * av.T) / ws_single_dim
        I_G = torch.sin(ws_single_dim * av.T) / ws_single_dim**2 - \
                av.T * torch.cos(ws_single_dim * av.T) / ws_single_dim
        logger.debug("hello 1")
        if isQuery:
            logger.debug("hello 2")
            thetas = ws_single_dim * embeds_single_dim
            logger.debug(f"thetas.shape: {thetas.shape}, R_G.shape: {R_G.shape}, I_G.shape: {I_G.shape}")
            fmap_single_dim_1 = torch.sign(R_G) * torch.sqrt(torch.abs(R_G)) * torch.cos(thetas) / sqrt_pdfs_single_dim
            fmap_single_dim_2 = torch.sign(R_G) * torch.sqrt(torch.abs(R_G)) * torch.sin(thetas) / sqrt_pdfs_single_dim
            fmap_single_dim_3 = - torch.sign(I_G) * torch.sqrt(torch.abs(I_G)) * torch.sin(thetas) / sqrt_pdfs_single_dim
            fmap_single_dim_4 = torch.sign(I_G) * torch.sqrt(torch.abs(I_G)) * torch.cos(thetas) / sqrt_pdfs_single_dim
        else:
            logger.debug("hello 3")
            thetas = ws_single_dim * embeds_single_dim
            logger.debug(f"thetas.shape: {thetas.shape}, R_G.shape: {R_G.shape}, I_G.shape: {I_G.shape}")
            fmap_single_dim_1 =  torch.sqrt(torch.abs(R_G)) * torch.cos(thetas) / sqrt_pdfs_single_dim
            fmap_single_dim_2 =  torch.sqrt(torch.abs(R_G)) * torch.sin(thetas) / sqrt_pdfs_single_dim
            fmap_single_dim_3 = torch.sqrt(torch.abs(I_G)) * torch.cos(thetas) / sqrt_pdfs_single_dim
            fmap_single_dim_4 =  torch.sqrt(torch.abs(I_G)) * torch.sin(thetas) / sqrt_pdfs_single_dim
        logger.debug("hello 4")
        fmap_single_dim[i] = torch.hstack((fmap_single_dim_1, fmap_single_dim_2, fmap_single_dim_3, fmap_single_dim_4))
    
    fmaps = torch.hstack(fmap_single_dim)
    logger.debug("hello 7")
    fmaps = fmaps.cpu().detach().numpy()
    #logger.debug("hello 8")
    #if not isQuery and not av.DEBUG:
    #    logger.info(f"Dumping corpus fourier maps to {fp}")
    #    all_d = {
    #        'fmaps': fmaps
    #    }
    #    pickle.dump(all_d, open(fp, "wb"))

    logger.debug(f'Fourier map generated for {type_}')
    return fmaps



class LSH(object):
    """
        Requires the following av args to be specified: 
        av.embed_dim  :       query/corpus expected to be of same (known) embedding dim
        av.m_use      :     decide when init
        av.hcode_dim :        decide when init
        av.num_hash_tables :  decide when init
        av.subset_size :      decode when init 
                              subsets of hashcode to consider in each hash function 
                              dictates no. of buckets
        We store following in LSH object: 
        w samples for feature maps: universal for all query/corpus points
        gaussian hplanes: (Universal) dictated by embedding dim, no of w samples ,required hashcode dim
    """
    def __init__(self, av): 
        super(LSH, self).__init__()
        self.av = av
        self.hashcode_pickle_fp = self.av.pickle_fp
        if self.hashcode_pickle_fp != "": 
            self.hdata = pickle.load(open(self.hashcode_pickle_fp, "rb"))
        self.num_hash_tables = self.av.num_hash_tables
        #No. of buckets in a hashTable is 2^subset_size
        self.subset_size = self.av.subset_size
        assert(self.subset_size<=av.hcode_dim)
        self.powers_of_two = cudavar(self.av,torch.from_numpy(1 << np.arange(self.subset_size - 1, -1, -1)).type(torch.FloatTensor))
        # generates self.hash_functions, containing indices for hash functions
        self.init_hash_functions()
        if self.av.use_pretrained_fmap: 
            self.gauss_hplanes_fhash = fetch_gaussian_hyperplanes(self.av, self.av.tr_fmap_dim)
        else:
            self.gauss_hplanes_fhash = fetch_gaussian_hyperplanes(self.av, self.av.embed_dim*self.av.m_use*4)
        if not self.av.FMAP_SELECT==-1:
            self.gauss_hplanes_fhash = self.gauss_hplanes_fhash[:self.av.FMAP_SELECT,:]

        if self.av.trained_cosine_fmap_pickle_fp != "" or self.av.use_pretrained_fmap: 
            self.gauss_hplanes_cos = fetch_gaussian_hyperplanes(self.av, self.av.tr_fmap_dim)
        else:
            self.gauss_hplanes_cos = fetch_gaussian_hyperplanes(self.av, self.av.embed_dim)
        self.gauss_hplanes_dot = fetch_gaussian_hyperplanes(self.av, self.av.embed_dim+1)
        self.gauss_hplanes_dot2 = fetch_gaussian_hyperplanes(self.av, self.av.embed_dim+2)
        self.ws, self.pdfs = fetch_samples(self.av)
        #print(self.ws.dtype, self.pdfs.dtype)
        assert(len(self.ws) == len(self.pdfs)) , f"Size mismatch{len(self.pdfs) }, {len(self.ws)}"
        assert(self.av.m_load == len(self.ws)) , f"Size mismatch{self.av.m_load }, {len(self.ws)}"
        self.sqrt_pdfs = np.sqrt(self.pdfs)
        self.R_G = 2 * (np.sin(self.ws * self.av.T/2))**2 / self.ws**2 + \
                self.av.T * np.sin(self.ws * self.av.T) / self.ws
        self.I_G = np.sin(self.ws * self.av.T) / self.ws**2 - \
                self.av.T * np.cos(self.ws * self.av.T) / self.ws
        self.sign_RG = np.sign(self.R_G)
        self.sign_IG = np.sign(self.I_G)
        self.sqrt_abs_RG = np.sqrt(np.abs(self.R_G))
        self.sqrt_abs_IG = np.sqrt(np.abs(self.I_G))

        self.temp1 = (self.sign_RG * self.sqrt_abs_RG) /self.sqrt_pdfs
        self.temp2 = (self.sign_IG * self.sqrt_abs_IG) / self.sqrt_pdfs
        self.temp3 = - self.temp2
        self.concat_temp = np.hstack([self.temp1, self.temp1, self.temp3, self.temp2])

        ### Timing methods anf funcs
        self.timing_methods = ["real", "user", "process_time"]
        self.timing_funcs = {"real": time.time, "user": user_time, "process_time": time.process_time}
        #self.mean = None
        self.trained_cosine_fmap_pickle_fp = self.av.trained_cosine_fmap_pickle_fp
        if self.trained_cosine_fmap_pickle_fp != "": 
            assert (self.av.pickle_fp == "")
            assert (self.av.HASH_MODE=="cosine")
            fmapdata = pickle.load(open(self.trained_cosine_fmap_pickle_fp, "rb"))
            self.hdata = {}
            batch_sz  = 50000
            #Writing split manually to ensure correctness
            batches = []
            for i in range(0, fmapdata['corpus'].shape[0],batch_sz):
                batches.append(fmapdata['corpus'][i:i+batch_sz])
            assert sum([item.shape[0] for item in batches]) == fmapdata['corpus'].shape[0]
            hcode_list = [] 
            for batch_item in batches :
                projections = batch_item@self.gauss_hplanes_cos
                hcode_list.append(cudavar(self.av,torch.tensor(np.sign(projections))))

            self.hdata['corpus'] =  torch.cat(hcode_list)
            self.hdata['query'] = {}
            #for mode in ["train", "test", "val"]:
            for mode in ["train", "val"]:
                projections = fmapdata['query'][mode].cpu()@self.gauss_hplanes_cos
                self.hdata['query'][mode] = cudavar(self.av,torch.tensor(np.sign(projections)))
        
        if self.av.use_pretrained_fmap: 
            self.tr_fmap_data,self.fmap_model_weights_np = check_pretrained_fmaps(self.av)
        if self.av.use_pretrained_hcode:
            #if self.av.TASK == "sighinge":
            #    self.tr_hcode_data,self.hcode_model_weights_np = check_pretrained_hashcodes_endtoend(self.av)
            #el
            if self.av.HASH_MODE == "cosine" or self.av.HASH_MODE == "dot" or self.av.HASH_MODE == "dot2" or self.av.HASH_MODE == "flora":
                self.tr_hcode_data,self.hcode_model_weights_np = check_pretrained_hashcodes_hingeemb(self.av)
            elif self.av.HASH_MODE == "fhash":
                self.tr_hcode_data,self.hcode_model_weights_np = check_pretrained_hashcodes(self.av)
            else:
                raise NotImplementedError()

        
    def generate_fmap(self, av, embeds, isQuery=False): 
        """
            Given some value of T, limit, a,b
            Fetch/generate prob samples
            compute map and return 
        """
        embeds_rep = np.repeat(embeds,av.m_use,axis=-1)
        #print(embeds_rep.dtype) 
        thetas = embeds_rep*self.ws
    
        cos_theta_by_sqrt_pdf = np.cos(thetas) / self.sqrt_pdfs
        sin_theta_by_sqrt_pdf = np.sin(thetas) / self.sqrt_pdfs
        if isQuery:
            fmap1 = self.sign_RG * self.sqrt_abs_RG * cos_theta_by_sqrt_pdf 
            fmap2 = self.sign_RG * self.sqrt_abs_RG * sin_theta_by_sqrt_pdf
            fmap3 = - self.sign_IG * self.sqrt_abs_IG * sin_theta_by_sqrt_pdf
            fmap4 = self.sign_IG * self.sqrt_abs_IG * cos_theta_by_sqrt_pdf
        else:
            fmap1 =  self.sqrt_abs_RG * cos_theta_by_sqrt_pdf
            fmap2 =  self.sqrt_abs_RG * sin_theta_by_sqrt_pdf
            fmap3 =  self.sqrt_abs_IG * cos_theta_by_sqrt_pdf
            fmap4 =  self.sqrt_abs_IG * sin_theta_by_sqrt_pdf
        
        fmaps = np.hstack([fmap1,fmap2,fmap3,fmap4])#.numpy()
      
        return fmaps
        
    def generate_query_fmap(self, av, embeds): 
        """
            Given some value of T, limit, a,b
            Fetch/generate prob samples
            compute map and return 
        """
        embeds_rep = np.repeat(embeds,av.m_use,axis=-1)
      
        thetas = embeds_rep*self.ws
    
        cos_theta = np.cos(thetas) 
        sin_theta = np.sin(thetas) 
            
        fmap1 = self.temp1 * cos_theta 
        fmap2 = self.temp1 * sin_theta
        fmap3 = self.temp3 * sin_theta
        fmap4 = self.temp2 * cos_theta
        
        fmaps = np.hstack([fmap1,fmap2,fmap3,fmap4])#.numpy()
      
        return fmaps

    def generate_query_fmap2(self, av, embeds): 
        """
            Given some value of T, limit, a,b
            Fetch/generate prob samples
            compute map and return 
        """
        thetas =  (embeds * self.ws.reshape(self.av.embed_dim,-1).T).T.flatten()
              
        cos_theta = np.cos(thetas) 
        sin_theta = np.sin(thetas) 
        
        fmaps = np.hstack([cos_theta,sin_theta,sin_theta,cos_theta])#,axis=-1)
        fmaps*= self.concat_temp
              
        return fmaps 

    def init_hash_functions(self):
        """
            Each hash function is a random subset of the hashcode. 
        """
        self.hash_functions = cudavar(self.av,torch.LongTensor([]))

        hash_code_dim = self.av.hcode_dim
        indices = list(range(hash_code_dim))
        for i in range(self.num_hash_tables):
            random.shuffle(indices)
            self.hash_functions= torch.cat((self.hash_functions,cudavar(self.av,torch.LongTensor([indices[:self.subset_size]]))),dim=0)


    def index_corpus(self, corpus_embeds,hash_mode):
        s = time.time()
        #TODO: may need to do torch conversion of corpus_embeds
        if self.av.HASH_MODE=="wmh":# and (corpus_embeds[0]==0).all():
            rows_idx = []
            for idx in range(len(corpus_embeds)):
                if all(corpus_embeds[idx]==0):
                    rows_idx.append(idx)
            self.corpus_embeds = np.delete(corpus_embeds,rows_idx,axis=0)
            #self.corpus_embeds = corpus_embeds[1:]
        else: 
            self.corpus_embeds = corpus_embeds

        assert self.av.embed_dim == self.corpus_embeds.shape[1]
        assert self.av.m_use * self.av.embed_dim <= self.av.m_load
        self.M = np.linalg.norm(corpus_embeds,axis=-1).max()

        if self.av.HASH_MODE=="wmh":
            self.corpus_hashcodes = get_wmh_hcode(self.av, self.corpus_embeds)
        else:
            self.corpus_hashcodes = self.fetch_RH_hashcodes(self.corpus_embeds,hash_mode,isQuery=False).cpu()
        assert(self.corpus_embeds.shape[0] == self.corpus_hashcodes.shape[0])
        self.num_corpus_items = self.corpus_embeds.shape[0]
        #generates self.hashcode_mat (containing +1/-1, used for bucketing)
        self.hashcode_mat = self.preprocess_hashcodes(self.corpus_hashcodes)
        #Assigns corpus items to buckets in each of the tables
        #generates dict self.all_hash_tables containing bucketId:courpusItemIDs
        self.bucketify()
        logger.info(f"Corpus indexed. Time taken {time.time()-s:.3f} sec")
        
        
    def fetch_RH_hashcodes(self, embeds, hash_mode, isQuery, qid=None):
        """
            embeds: can be 1 or many from query or corpus
        """
        if self.hashcode_pickle_fp != "" or self.trained_cosine_fmap_pickle_fp != "":
            if isQuery == False:
                #print(f"isQuery: {isQuery}  hashcodes shape : {self.hdata['corpus'].double().shape} {self.hdata['corpus'].double().dtype}")
                return cudavar(self.av,self.hdata['corpus'].double())
            else: 
                #print(f"isQuery: {isQuery}  hashcodes shape : {self.hdata['query'][self.av.SPLIT][qid].double()[None,:].shape} {self.hdata['query'][self.av.SPLIT][qid].double()[None,:].dtype}")
                return cudavar(self.av,self.hdata['query'][self.av.SPLIT][qid].double()[None,:])

        if self.av.SPECIAL_FHASH == "SymCorpus":
            isQuery = False
        elif self.av.SPECIAL_FHASH == "SymQuery":
            isQuery = True
        max_norm = np.max(np.linalg.norm(embeds, axis=1))
        d = embeds.shape[1]
        batch_sz  = 50000
        #Writing split manually to ensure correctness
        batches = []
        for i in range(0, embeds.shape[0],batch_sz):
            batches.append(embeds[i:i+batch_sz])
        assert sum([item.shape[0] for item in batches]) == embeds.shape[0]
        hcode_list = [] 
        for batch_item in batches :
            if hash_mode == "cosine":
                if self.av.use_pretrained_hcode:
                    #print("Should be here")
                    if self.hcode_model_weights_np['num_layers']==1:
                        projections = batch_item@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                        projections = np.tanh(self.av.TANH_TEMP * projections)
                    else:
                        projections = batch_item@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                        for idx in range(1,self.hcode_model_weights_np['num_layers']):
                            projections[projections<0]=0
                            projections = projections@self.hcode_model_weights_np[idx]['np_w'].T+self.hcode_model_weights_np[idx]['np_b']
                        projections = np.tanh(self.av.TANH_TEMP * projections)
                else:    
                    #print("Should NOT be here")
                    projections = batch_item@self.gauss_hplanes_cos
                hcode_list.append(cudavar(self.av,torch.tensor(np.sign(projections))))
            elif hash_mode == "flora":
                #if self.av.use_pretrained_hcode:
                if isQuery:
                    fmaps = batch_item@self.hcode_model_weights_np['np_w_q'].T+self.hcode_model_weights_np['np_b_q']
                else:
                    fmaps = batch_item@self.hcode_model_weights_np['np_w_c'].T+self.hcode_model_weights_np['np_b_c']
                assert self.hcode_model_weights_np['num_layers']==1
                projections = fmaps@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                projections = np.tanh(self.av.TANH_TEMP * projections)
                #else:    
                #    #print("Should NOT be here")
                #    projections = batch_item@self.gauss_hplanes_cos
                hcode_list.append(cudavar(self.av,torch.tensor(np.sign(projections))))
            elif hash_mode == "dot" or hash_mode == "dot2":
                #NOTE: for query it will ultimately append 0, but we may directly do that under if isQuery to speed up
                if hash_mode == "dot":
                    batch_item_scaled = batch_item/max_norm
                    app = np.expand_dims(np.sqrt(1-np.square(np.linalg.norm(batch_item_scaled, axis=1))),axis=-1)
                    batch_item_augmented = np.hstack((batch_item_scaled,app))
                elif hash_mode == "dot2":
                    zero_np = np.zeros((batch_item.shape[0],1))
                    app = np.expand_dims(np.sqrt(np.square(self.M)-np.square(np.linalg.norm(batch_item, axis=1))),axis=-1)
                    if isQuery:
                        batch_item_augmented = np.hstack((app,batch_item,zero_np))/self.M
                    else:
                        batch_item_augmented = np.hstack((zero_np,batch_item,app))/self.M
                        
                
                if self.av.use_pretrained_hcode:
                    if self.hcode_model_weights_np['num_layers']==1:
                        projections = batch_item_augmented@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                        projections = np.tanh(self.av.TANH_TEMP * projections)
                    else:
                        projections = batch_item_augmented@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                        for idx in range(1,self.hcode_model_weights_np['num_layers']):
                            projections[projections<0]=0
                            projections = projections@self.hcode_model_weights_np[idx]['np_w'].T+self.hcode_model_weights_np[idx]['np_b']
                        projections = np.tanh(self.av.TANH_TEMP * projections)
                else:    
                    if hash_mode == "dot":
                        projections = batch_item_augmented@self.gauss_hplanes_dot
                    elif  hash_mode == "dot2":
                        projections = batch_item_augmented@self.gauss_hplanes_dot2

                hcode_list.append(cudavar(self.av,torch.tensor(np.sign(projections))))
            elif hash_mode == "fhash":
                #fmaps =  generate_fourier_map(self.av, batch_item, self.ws, self.pdfs, isQuery)
                #fmaps =  generate_fourier_map_tensorized_np(self.av, batch_item, self.ws[:d*self.av.m_use], self.sqrt_pdfs[:d*self.av.m_use], isQuery)
                #if isQuery: 
                #    fmaps =  self.generate_query_fmap2(self.av, batch_item/self.av.SCALE)[None,:]
                #else:
                fmaps =  self.generate_fmap(self.av, batch_item/self.av.SCALE, isQuery)
                if not self.av.FMAP_SELECT==-1:
                    fmaps = fmaps[:,:self.av.FMAP_SELECT]
                if self.av.use_pretrained_fmap:
                    if self.av.FMAP_LOSS_TYPE == "AsymFmapCos":
                        if isQuery:
                            fmaps = fmaps@self.fmap_model_weights_np['np_w_q'].T+self.fmap_model_weights_np['np_b_q']
                            fmaps = fmaps/np.linalg.norm(fmaps,axis=-1,keepdims=True)
                            #print("Should be here Query: ", qid)
                            #assert (np.allclose(fmaps,self.tr_fmap_data['query'][self.av.SPLIT].cpu().numpy()[qid],atol=1e-06))
                        else:
                            fmaps = fmaps@self.fmap_model_weights_np['np_w_c'].T+self.fmap_model_weights_np['np_b_c']
                            fmaps = fmaps/np.linalg.norm(fmaps,axis=-1,keepdims=True)
                            #print("Should be here", fmaps.shape)
                    elif self.av.FMAP_LOSS_TYPE == "FmapCos":
                        fmaps = fmaps@self.fmap_model_weights_np['np_w'].T+self.fmap_model_weights_np['np_b']
                        fmaps = fmaps/np.linalg.norm(fmaps,axis=-1,keepdims=True)

                    
                
                if self.av.DEBUG:
                    assert np.all(np.isclose(np.linalg.norm(fmaps[0]),\
                                             np.linalg.norm(fmaps, axis=1)))

                assert fmaps.shape[0] == batch_item.shape[0],\
                f"fmaps shape={fmaps.shape[0]}, batch_item shape={batch_item.shape[0]}"
                #if self.mean is None:                
                #    self.mean =  np.mean(fmaps,axis=0)
                #fmaps -= self.mean
                if self.av.use_pretrained_hcode:
                    if  self.av.E2EVER=="e2e3" or self.av.E2EVER=="e2e4":
                        if self.hcode_model_weights_np['num_layers']==1:
                            projections = batch_item@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                            projections = np.tanh(self.av.TANH_TEMP * projections)
                        else:
                            projections = batch_item@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                            for idx in range(1,self.hcode_model_weights_np['num_layers']):
                                projections[projections<0]=0
                                projections = projections@self.hcode_model_weights_np[idx]['np_w'].T+self.hcode_model_weights_np[idx]['np_b']
                            projections = np.tanh(self.av.TANH_TEMP * projections)
                        #projections = batch_item@self.hcode_model_weights_np['np_w'].T+self.hcode_model_weights_np['np_b']
                        #projections = np.tanh(self.av.TANH_TEMP * projections)
                    else:
                        if self.hcode_model_weights_np['num_layers']==1:
                            projections = fmaps@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                            projections = np.tanh(self.av.TANH_TEMP * projections)
                        else:
                            projections = fmaps@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
                            for idx in range(1,self.hcode_model_weights_np['num_layers']):
                                projections[projections<0]=0
                                projections = projections@self.hcode_model_weights_np[idx]['np_w'].T+self.hcode_model_weights_np[idx]['np_b']
                            projections = np.tanh(self.av.TANH_TEMP * projections)
                        #print("Should be here")
                        #projections = fmaps@self.hcode_model_weights_np['np_w'].T+self.hcode_model_weights_np['np_b']
                        #projections = np.tanh(self.av.TANH_TEMP * projections)
                        #print("Shoulf not be here")
                else:    
                    #print("Should be here")
                    projections = fmaps@self.gauss_hplanes_fhash
                
                hcode_list.append(cudavar(self.av,torch.tensor(np.sign(projections))))
            elif hash_mode == "random":
                random_hash_codes = np.random.choice([-1,1], size=(batch_item.shape[0], self.av.hcode_dim))
                hcode_list.append(cudavar(self.av,torch.tensor(random_hash_codes, dtype=float)))
            elif hash_mode == "fhash_sym":
                fmaps =  self.generate_fmap(self.av, batch_item/self.av.SCALE, True)
                assert fmaps.shape[0] == batch_item.shape[0],\
                f"fmaps shape={fmaps.shape[0]}, batch_item shape={batch_item.shape[0]}"
                projections = fmaps@self.gauss_hplanes_fhash
                hcode_list.append(cudavar(self.av,torch.tensor(np.sign(projections))))
            else:
                #Can be thrown if hash_mode=None for no_bucket and somehow hashcodes are being generated for query
                raise NotImplementedError()
        hashcodes = torch.cat(hcode_list)
        #print(f"isQuery: {isQuery}  hashcodes shape : {hashcodes.shape}  {hashcodes.dtype} ")
        return hashcodes


    def preprocess_hashcodes(self,all_hashcodes): 
        if self.av.HASH_MODE=="wmh":
            return all_hashcodes
        all_hashcodes = cudavar(self.av,torch.sign(all_hashcodes))
        if (torch.sign(all_hashcodes)==0).any(): 
            logger.info("Hashcode had 0 bits. replacing all with 1")
            all_hashcodes[all_hashcodes==0]=1
        return all_hashcodes


    def assign_bucket(self,function_id,node_hash_code):
        func = self.hash_functions[function_id]
        
        if self.av.HASH_MODE=="wmh":
            return '_'.join(node_hash_code[func])
        
        # convert sequence of -1 and 1 to binary by replacing -1 s to 0
        binary_id = torch.max(torch.index_select(node_hash_code,dim=0,index=func),cudavar(self.av,torch.LongTensor([0])))
        #map binary sequence to int which is bucket Id
        bucket_id = self.powers_of_two@binary_id.type(torch.FloatTensor).to(self.powers_of_two)
        return bucket_id.item()

    def bucketify(self): 
        """
          For all hash functions: 
            Loop over all corpus items
              Assign corpus item to bucket in hash table corr. to hash function 
        """ 
        s = time.time()
        self.all_hash_tables = []
        for func_id in range(self.num_hash_tables): 
            hash_table = defaultdict(list)#{}
            #for idx in range(2**self.subset_size): 
            #    hash_table[idx] = []
            for item in range(self.num_corpus_items):
                hash_table[self.assign_bucket(func_id,self.hashcode_mat[item])].append(item)
            self.all_hash_tables.append(hash_table)
    
    def pretty_print_hash_tables(self,topk): 
        """
            I've found this function useful to visualize corpus distribution across buckets
        """
        for table_id in range(self.num_hash_tables): 
            if self.av.HASH_MODE=="wmh":
                len_list = sorted([len(self.all_hash_tables[table_id][bucket_id]) for bucket_id in self.all_hash_tables[table_id].keys()])[::-1] [:topk]
            else:
                len_list = sorted([len(self.all_hash_tables[table_id][bucket_id]) for bucket_id in range(2**self.subset_size)])[::-1] [:topk]
            len_list_str = [str(i) for i in len_list]
            lens = '|'.join(len_list_str)
            print(lens)

    def set_flora_info(self,qembeds,cembeds):
        self.qembeds= qembeds
        self.cembeds= cembeds

    def heapify (self,q_embed,candidate_list, K, use_tensor=False,qid=0):
        """
            use q_embed , candidate_list, corpus_embeds to fetch top K items
        """
        time_dict = {tm: {} for tm in self.timing_methods}
        other_data_dict = {}
        #if use_tensor then parallelize scoring, else GPU
        #TODO: discuss if we want to do the GPU vs non-GPU thing
        #If no, then torch can be avoided completely
        #If yes, we need to use torch instead of np below
        if len(candidate_list) == 0:
            for k in time_dict.keys():
                time_dict[k]['score_computation_time'] = 0.0
                time_dict[k]['heap_procedure_time'] = 0.0
                time_dict[k]['take_time'] = 0.0
                time_dict[k]['sort_procedure_time'] = 0.0
            other_data_dict['len_candidate_list'] = 0
            return list(), list(), time_dict, other_data_dict
        else:
            #print(len(candidate_list),self.corpus_embeds.shape)
            score_timer_start = {}
            for tm in self.timing_methods:
                score_timer_start[tm] = self.timing_funcs[tm]()

            #s = time.time()
            take_start = {}
            for tm in self.timing_methods:
                take_start[tm] = self.timing_funcs[tm]()
            candidate_corpus_embeds = np.take(self.corpus_embeds,candidate_list,axis=0)
            for tm in self.timing_methods:
                time_dict[tm]['take_time'] = self.timing_funcs[tm]() - take_start[tm]
            #print("DEBUG score::",time.time()-s)
            #s = time.time()
            if self.av.HASH_MODE == 'flora':
                candidate_corpus_embeds = np.take(self.cembeds,candidate_list,axis=0)
                #scores = asym_sim(self.av, self.qembeds[qid],candidate_corpus_embeds) 
                scores = sigmoid_asym_sim(self.av, self.qembeds[qid],candidate_corpus_embeds)
            elif self.av.TASK == 'hinge':
                # self.pretty_print_hash_tables(15)
                # exit(1)
                scores = asym_sim(self.av, q_embed,candidate_corpus_embeds) 
            elif self.av.TASK == 'sighinge':
                # self.pretty_print_hash_tables(15)
                # exit(1)
                scores = sigmoid_asym_sim(self.av, q_embed,candidate_corpus_embeds) 
            elif self.av.TASK == 'dot':
                # self.pretty_print_hash_tables(15)
                # exit(1)
                scores = dot_sim(q_embed,candidate_corpus_embeds).reshape(-1) 
            elif self.av.TASK == 'wjac':
                # self.pretty_print_hash_tables(15)
                # exit(1)
                scores = wjac_sim(q_embed,candidate_corpus_embeds).reshape(-1) 
            else:
                # self.pretty_print_hash_tables(15)
                # exit(1)
                scores = optimized_cosine_similarity(q_embed,candidate_corpus_embeds).reshape(-1)
            #print("DEBUG score::",time.time()-s)
            for tm in self.timing_methods:
                time_dict[tm]['score_computation_time'] = self.timing_funcs[tm]() - score_timer_start[tm]
            heap_timer_start = {}
            for tm in self.timing_methods:
                heap_timer_start[tm] = self.timing_funcs[tm]()
            #s = time.time()
            if K >= 0:    
                score_heap = []
                heap_size = 0

                #print("DEBUG heap::",time.time()-s)
                #s = time.time()

                for i in range(len(candidate_list)):
                    if heap_size<K: 
                        heap_size = heap_size+1
                        heapq.heappush(score_heap,(scores[i],candidate_list[i]))
                    else:
                        heapq.heappushpop(score_heap,(scores[i],candidate_list[i]))

                #print("DEBUG heap::",time.time()-s)
                for tm in self.timing_methods:
                    time_dict[tm]['heap_procedure_time'] = self.timing_funcs[tm]() - heap_timer_start[tm]
                scores,corpus_ids =  list(zip (*score_heap))

            else:
                corpus_ids = candidate_list
                for tm in self.timing_methods:
                    time_dict[tm]['heap_procedure_time'] = self.timing_funcs[tm]() - heap_timer_start[tm]
            sort_timer_start = {}
            for tm in self.timing_methods:
                sort_timer_start[tm] = self.timing_funcs[tm]()
            # sort_zip_timer_start = time.time()
            # score_with_corpus_ids = list(zip(scores, corpus_ids))
            # time_dict['sort_zip_time'] = time.time() - sort_zip_timer_start
            # sorted_score_with_corpus_ids = sorted(score_with_corpus_ids, reverse=True)
            # sort_unzip_timer_start = time.time()
            # sorted_scores, sorted_corpus_ids = list(zip(*sorted_score_with_corpus_ids))
            # time_dict['sort_unzip_time'] = time.time() - sort_unzip_timer_start
            scores_arr = np.array(scores)
            corpus_ids_arr = np.array(corpus_ids)
            sorted_ids = np.argsort(-scores_arr)
            sorted_scores = scores_arr[sorted_ids]
            sorted_corpus_ids = corpus_ids_arr[sorted_ids]
            for tm in self.timing_methods:
                time_dict[tm]['sort_procedure_time'] = self.timing_funcs[tm]() - sort_timer_start[tm]
            other_data_dict['len_candidate_list'] = len(sorted_corpus_ids)
            return list(sorted_scores), list(sorted_corpus_ids), time_dict, other_data_dict


    def retrieve(self,q_embed,K,hash_mode=None,no_bucket=False,qid=None): 
        """
            Input : query_embed : to compute actual asym dist 
                      shape is (1*d)
            Input : K : top K similar items to return
            Output : top K items, time taken for retrieval, accuracy? 

            given query and a number k, find the top k closest corpus items 
            loop over al hash_tables: 
              map query to corr bucket: 
                compute Asymmetric similarity between query and each corpus item in bucket and update min heap
            return [TODO: decide metric] of corpus in min heap
            TODO: guard against too few corpus items? 
        """
        #Given input query_embed: generate query_hashcode : to ID query bucket 
        start_hashcode_gen = {}
        end_hashcode_gen = {}
        if not no_bucket: 
            for tm in self.timing_methods:
                start_hashcode_gen[tm] = self.timing_funcs[tm]()

            if self.av.HASH_MODE=="wmh":
                #WMH(DrHash) toolkit throws exception for zero vectors. Adding small random noise
                if all(q_embed[0]==0):
                    q_embed = q_embed + np.random.normal(0,1e-8,len(q_embed[0]))
                q_hashcode = get_wmh_hcode(self.av, q_embed)
            else:
                q_hashcode =  self.preprocess_hashcodes(self.fetch_RH_hashcodes(q_embed,hash_mode,isQuery=True,qid=qid)).squeeze().cpu()
            
            for tm in self.timing_methods:
                end_hashcode_gen[tm] = self.timing_funcs[tm]()
        else:
            for tm in self.timing_methods:
                start_hashcode_gen[tm] = 0
                end_hashcode_gen[tm] = 0
        
        start_candidate_list_gen = {}
        end_candidate_list_gen = {}
        if no_bucket:
            for tm in self.timing_methods:
                start_candidate_list_gen[tm] = self.timing_funcs[tm]() 
            #We consider all corpus items  
            candidate_list = list(range(self.num_corpus_items))
            for tm in self.timing_methods:
                end_candidate_list_gen[tm] = self.timing_funcs[tm]()
        else:
            for tm in self.timing_methods:
                start_candidate_list_gen[tm] = self.timing_funcs[tm]()
            #We use q hashcode to identify buckets, and take union of corpus items into candidate set
            candidate_list = []
            for table_id in range(self.num_hash_tables): 
                #identify bucket 
                bucket_id = self.assign_bucket(table_id,q_hashcode)
                candidate_list.extend(self.all_hash_tables[table_id][bucket_id])

            #remove duplicates from candidate_list
            candidate_list = list(set(candidate_list))
            for tm in self.timing_methods:
                end_candidate_list_gen[tm] = self.timing_funcs[tm]()

            if self.av.DEBUG:
                print("No. of candidates found", len(candidate_list))

        if hash_mode == "flora":
            scores, corpus_ids, time_dict, other_data_dict = self.heapify (q_embed,candidate_list, K, use_tensor=False,qid=qid)
        else:
            scores, corpus_ids, time_dict, other_data_dict = self.heapify (q_embed,candidate_list, K, use_tensor=False)
        for tm in self.timing_methods:
            time_dict[tm]['candidate_list_gen_time'] = end_candidate_list_gen[tm] - start_candidate_list_gen[tm]  
            time_dict[tm]['hashcode_gen_time'] = end_hashcode_gen[tm] - start_hashcode_gen[tm]
        return len(candidate_list),  scores,corpus_ids, time_dict, other_data_dict



