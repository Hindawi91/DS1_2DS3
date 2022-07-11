from PIL import Image
import os, sys
folders = ["train","test"]
sub_folders = ["negative","positive"]

for i,folder in enumerate(folders):
    for sub in sub_folders:
        
        path = f"/home/local/ASUAD/falhinda/Downloads/Fixed-Point-GAN-3/data/brats/syn/{folder}/{sub}/"
        output = f"/home/local/ASUAD/falhinda/Downloads/Fixed-Point-GAN-3/data/brats/syn2/{folder}/{sub}/"
        if not os.path.exists(output):
            os.makedirs(output)
        
    
        dirs = os.listdir( path )
        
        def resize():
            for item in dirs:
                if os.path.isfile(path+item):
                    im = Image.open(path+item)
                    f, e = os.path.splitext(path+item)
                    imResize = im.resize((500,500), Image.ANTIALIAS)
                    imResize.save(output + item, 'JPEG', quality=100)
        
        resize()
