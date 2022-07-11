import cv2
from keras.preprocessing import image
from glob import glob                                                           
import pandas as pd
import datetime
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
 
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for img in images:
		# resize with nearest neighbor interpolation
		new_image = resize(img, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
    
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))


real_imgs=[]


real_files = glob('./data/brats/syn/train/negative/*.jpg*')

for i,im_file in enumerate (real_files):
    
    img1 = image.load_img(im_file,target_size=(256, 256))
    img1 = image.img_to_array(img1)
    img1 /= 255.
    real_imgs.append(img1)
    
real_imgs = numpy.array(real_imgs)
    
All_Models = []
All_Metrics = []
All_CMs = []


expirements = ["exp1"]

# expirements = ["exp1"]
for exp in range(10000,300001,10000):
    begin_time = datetime.datetime.now()
    fake_imgs=[]
    fake_files = glob(f'./brats_syn_256_lambda0.1/results_{exp}/*.jpg*')
    for i,im_file in enumerate (fake_files):
        
        img1 = image.load_img(im_file,target_size=(256, 256))
        img1 = image.img_to_array(img1)
        img1 /= 255.
        fake_imgs.append(img1)   
            
    print('real file 10 test: ',real_files[10])
    print('fake file 10 test: ',fake_files[10])
    fake_imgs = numpy.array(fake_imgs)   
    
    print('Prepared', real_imgs.shape, fake_imgs.shape)
    # convert integer to floating point values
    real_imgs = real_imgs.astype('float32')
    fake_imgs = fake_imgs.astype('float32')
    # resize images
    real_imgs = scale_images(real_imgs, (299,299,3))
    fake_imgs = scale_images(fake_imgs, (299,299,3))
    print('Scaled', real_imgs.shape, fake_imgs.shape)
    # pre-process images
    real_imgs = preprocess_input(real_imgs)
    fake_imgs = preprocess_input(fake_imgs)
    # fid between images1 and images1
    real_vs_real_fid = calculate_fid(model, real_imgs, real_imgs)
    print('FID (same): %.3f' % real_vs_real_fid)
    # fid between images1 and images2
    fake_vs_real_fid = calculate_fid(model, real_imgs, fake_imgs)
    print('FID (different): %.3f' % fake_vs_real_fid)
    
    testing_time = datetime.datetime.now() - begin_time
    
    metrics = [exp,real_vs_real_fid,fake_vs_real_fid,testing_time]
    metrics_names = ["GAN Model","real_vs_real_fid","fake_vs_real_fid","testing_time"]

    All_Metrics.append(metrics)


df = pd.DataFrame(All_Metrics,columns=metrics_names)

df.to_excel ('./Val - FID_Metrics.xlsx', index = False, header=True)









