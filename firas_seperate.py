import shutil
import os

catgeories = ["ONB","BIC","CHF"]

src_dir = f"/home/local/ASUAD/falhinda/Downloads/Fixed-Point-GAN-2/brats_syn_256_lambda0.1/results"

GIT_dir = f"/home/local/ASUAD/falhinda/Downloads/Fixed-Point-GAN-2/brats_syn_256_lambda0.1/results_formatted/GIT"
UT_dir = f"/home/local/ASUAD/falhinda/Downloads/Fixed-Point-GAN-2/brats_syn_256_lambda0.1/results_formatted/UT"



file_names = os.listdir(src_dir)
print(len(file_names))


for file_name in file_names:
    
    if "GIT" in file_name:
        for category in catgeories:
            if category in file_name:
                sub_dir = f'{GIT_dir}/{category}'
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                shutil.copy(os.path.join(src_dir, file_name), sub_dir)
                print(f'copying {file_name} from {src_dir} to {sub_dir}')
        
    else:
        
        for category in catgeories:
            if category in file_name:
                sub_dir = f'{UT_dir}/{category}'
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                shutil.copy(os.path.join(src_dir, file_name), sub_dir)
