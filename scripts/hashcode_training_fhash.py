import os
import subprocess

dev = "0"
margin_list = [0.05]
variations = [(10,"Asym","AsymFmapCos")]

c1_val = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]
scloss_margin = 1.0


with open("./curr_comms.sh", "w") as text_file:
        text_file.write("#!/bin/bash")
        text_file.write("\n")

for margin in margin_list:
    for v1,v2,v3 in variations:
        #for sc_ss in [8,9,10]:
        for sc_ss in [8]:
            #for fmaptype in ["BCE3","MSE","BCE"]:
            for fmaptype in ["BCE3"]:
                for dval in c1_val:
                    #command_str = f"CUDA_VISIBLE_DEVICES={dev} python -m src.hashcode_trainer1 --DATASET_NAME=\"syn\" --m_use=10 --m_load=200 --T=38 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS  --embed_dim=20 --TASK=\"sighinge\" --HASH_MODE=\"cosine\" --num_hash_tables=10 --ES=50 --LOSS_TYPE=\"sc_loss\" --sc_subset_size={sc_ss} --TANH_TEMP=1  --tr_fmap_dim={v1}  --FMAP_LOSS_TYPE={v3} --MARGIN={margin}  --FENCE_LAMBDA=0.1 --C1_LAMBDA={dval} --WEAKSUP_LAMBDA=0 --pickle_fp=\"\" --FMAP_SELECT=-1 --SPECIAL_FHASH={v2} --trained_cosine_fmap_pickle_fp=\"./data/fmapPickles/sighinge_syn_MARGIN{margin}_muse10_T38.0_Scale1_TrFmapDim{v1}_fmapSelect_-1_SpecialFhash_{v2}_LOSS_{v3}_arch_L_fmap{fmaptype}_fmap_mat.pkl\" --LEARNING_RATE=1e-3 --SCLOSS_MARGIN={scloss_margin} --USE_FMAP_{fmaptype}"
                    #command_str = f"CUDA_VISIBLE_DEVICES={dev} python -m src.hashcode_trainer1 --DATASET_NAME=\"msnbc294_4\" --m_use=10 --m_load=2940 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS  --embed_dim=294 --TASK=\"sighinge\" --HASH_MODE=\"cosine\" --num_hash_tables=10 --ES=50 --LOSS_TYPE=\"permgnn_loss\" --sc_subset_size={sc_ss} --TANH_TEMP=1  --tr_fmap_dim={v1}  --FMAP_LOSS_TYPE={v3} --MARGIN={margin}  --FENCE_LAMBDA=0.1 --DECORR_LAMBDA={dval} --WEAKSUP_LAMBDA=0 --pickle_fp=\"\" --FMAP_SELECT=-1 --SPECIAL_FHASH={v2} --trained_cosine_fmap_pickle_fp=\"./data/fmapPickles/sighinge_msnbc294_4_MARGIN{margin}_muse10_T3.0_Scale1_TrFmapDim{v1}_fmapSelect_-1_SpecialFhash_{v2}_LOSS_{v3}_arch_L_fmap{fmaptype}_fmap_mat.pkl\" --LEARNING_RATE=1e-3 --SCLOSS_MARGIN={scloss_margin} --USE_FMAP_{fmaptype}"
                    command_str = f"CUDA_VISIBLE_DEVICES={dev} python -m src.hashcode_trainer1 --DATASET_NAME=\"msweb294\" --m_use=5 --m_load=1470 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS  --embed_dim=294 --TASK=\"sighinge\" --HASH_MODE=\"cosine\" --num_hash_tables=10 --ES=50 --LOSS_TYPE=\"sc_loss\" --sc_subset_size={sc_ss} --TANH_TEMP=1  --tr_fmap_dim={v1}  --FMAP_LOSS_TYPE={v3} --MARGIN={margin}  --FENCE_LAMBDA=0.1 --C1_LAMBDA={dval} --WEAKSUP_LAMBDA=0 --pickle_fp=\"\" --FMAP_SELECT=-1 --SPECIAL_FHASH={v2} --trained_cosine_fmap_pickle_fp=\"./data/fmapPickles/sighinge_msweb294_MARGIN{margin}_muse5_T3.0_Scale1_TrFmapDim{v1}_fmapSelect_-1_SpecialFhash_{v2}_LOSS_{v3}_arch_L_fmap{fmaptype}_fmap_mat.pkl\" --LEARNING_RATE=1e-3 --SCLOSS_MARGIN={scloss_margin} --USE_FMAP_{fmaptype}"
                    #command_str = f"CUDA_VISIBLE_DEVICES={dev} python -m src.hashcode_trainer1 --DATASET_NAME=\"msweb294\" --m_use=50 --m_load=14700 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS  --embed_dim=294 --TASK=\"sighinge\" --HASH_MODE=\"cosine\" --num_hash_tables=10 --ES=50 --LOSS_TYPE=\"sc_loss\" --sc_subset_size={sc_ss} --TANH_TEMP=1  --tr_fmap_dim={v1}  --FMAP_LOSS_TYPE={v3} --MARGIN={margin}  --FENCE_LAMBDA=0.1 --C1_LAMBDA={dval} --WEAKSUP_LAMBDA=0 --pickle_fp=\"\" --FMAP_SELECT=-1 --SPECIAL_FHASH={v2} --trained_cosine_fmap_pickle_fp=\"./data/fmapPickles/sighinge_msweb294_MARGIN{margin}_muse50_T3.0_Scale1_TrFmapDim{v1}_fmapSelect_-1_SpecialFhash_{v2}_LOSS_{v3}_arch_L_fmap{fmaptype}_fmap_mat.pkl\" --LEARNING_RATE=1e-3 --SCLOSS_MARGIN={scloss_margin} --USE_FMAP_{fmaptype}"
                    with open("./curr_comms.sh", "a") as text_file:
                            text_file.write("(")
                            text_file.write(command_str)
                            text_file.write(") & \n")
                    #subprocess.call(["chmod", "+x", "run_sc6.sh"])
                    #subprocess.call(["sh","./run_sc6.sh"])
                    #subprocess.call(["rm", "-f" ,"run_sc6.sh"])
