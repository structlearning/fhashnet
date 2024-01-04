import os
import subprocess

dev = "1"
c1_val = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]
scloss_margin = 1.0

with open("./curr_comms.sh", "w") as text_file:
        text_file.write("#!/bin/bash")
        text_file.write("\n")


for dval in c1_val:
    #command_str = f"CUDA_VISIBLE_DEVICES={dev} python -m src.hashcode_trainer_hingeemb --DATASET_NAME=\"syn\" --m_use=10 --m_load=200 --embed_dim=20 --T=38 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS  --embed_dim=20 --TASK=\"sighinge\" --HASH_MODE=\"dot2\" --num_hash_tables=10 --ES=50 --LOSS_TYPE=\"sc_loss_hingeemb\" --sc_subset_size=8 --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --C1_LAMBDA={dval} --WEAKSUP_LAMBDA=0 --pickle_fp=\"\" --FMAP_SELECT=-1 --SPECIAL_FHASH=\"\" --trained_cosine_fmap_pickle_fp=\"\" --LEARNING_RATE=1e-3 --SCLOSS_MARGIN={scloss_margin}"
    #command_str = f"CUDA_VISIBLE_DEVICES={dev} python -m src.hashcode_trainer_hingeemb --DATASET_NAME=\"msnbc294_3\" --m_use=10 --m_load=2940 --embed_dim=294 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS  --embed_dim=294 --TASK=\"dot\" --HASH_MODE=\"dot\" --num_hash_tables=10 --ES=50 --LOSS_TYPE=\"sc_loss_hingeemb\" --sc_subset_size=8 --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --C1_LAMBDA={dval} --WEAKSUP_LAMBDA=0 --pickle_fp=\"\" --FMAP_SELECT=-1 --SPECIAL_FHASH=\"\" --trained_cosine_fmap_pickle_fp=\"\" --LEARNING_RATE=1e-3 --SCLOSS_MARGIN={scloss_margin}"
    command_str = f"CUDA_VISIBLE_DEVICES={dev} python -m src.hashcode_trainer_hingeemb --DATASET_NAME=\"msnbc294_5\" --m_use=10 --m_load=2940 --embed_dim=294 --T=3 --SCALE=1 --hcode_dim=64 --HIDDEN_LAYERS  --embed_dim=294 --TASK=\"sighinge\" --HASH_MODE=\"cosine\" --num_hash_tables=10 --ES=50 --LOSS_TYPE=\"permgnn_loss_hingeemb\" --sc_subset_size=8 --TANH_TEMP=1 --FENCE_LAMBDA=0.1 --DECORR_LAMBDA={dval} --WEAKSUP_LAMBDA=0 --pickle_fp=\"\" --FMAP_SELECT=-1 --SPECIAL_FHASH=\"\" --trained_cosine_fmap_pickle_fp=\"\" --LEARNING_RATE=1e-3 --SCLOSS_MARGIN={scloss_margin}"
    with open("./curr_comms.sh", "a") as text_file:
        text_file.write("(")
        text_file.write(command_str)
        text_file.write(") & \n")

    #subprocess.call(["chmod", "+x", "run_cos_hingescloss1.sh"])
    #subprocess.call(["sh","./run_cos_hingescloss1.sh"])
    #subprocess.call(["rm", "-f" ,"run_cos_hingescloss1.sh"])


