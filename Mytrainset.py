import numpy as np
from multiprocessing import Pool
import healpy as hp
from pygdsm import GlobalSkyModel16
from random import randrange as rr
import matplotlib.pyplot as plt
from astropy.io import fits
input_path = "/scr1/nulee/SDC3"
save_path = '/scr1/nulee/SDC3/trainset'
beamsize = 4000

'''first commit
def MJysr_to_Jybeam(mjysr, beamsize_arcsec2):
    beamsize_sr = beamsize_arcsec2 * (np.pi / (180**2 * 3600**2))
    jysr = mjysr * 1e6
    jybeam = jysr * beamsize_sr
    return jybeam

background=np.zeros((264, 2531, 5062))
gsm_2016 = GlobalSkyModel16(freq_unit='MHz',data_unit='MJysr', resolution='hi')
for d in range(264):
    freq = 106 + 0.1 * d
    cart = gsm_2016.generate(freq)
    temp = hp.cartview(cart, xsize=5062, ysize=2531, return_projected_map=True, hold=False,fig=1).data
    temp -= np.mean(temp)
    temp = MJysr_to_Jybeam(temp, beamsize)
    background[d, : ,: ]=temp
    plt.close('all')
np.save(input_path + '/background',background.astype(np.float32))
print("background_generated")
'''

'''Saving Real SDC3 data
SDC3 = fits.open("/vega/home/nulee/ZW3.msw_image.fits")[0].data
for i in range(100):
    i, j, k = rr(64, 200), rr(600, 1448), rr(600, 1448)
    SDC3_real = SDC3[i-64:i+64,j-64:j+64,k-64:k+64].astype(np.float32)
np.save(save_path + '/SDC3_REAL',SDC3_real.astype(np.float32))
'''
def point_sources(path,i,j,k):
    ps=np.load(path+"/ps.npy")
    return ps[i-64:i+64,j-64:j+64,k-64:k+64].astype(np.float32)

def Augmentation(input_trainset):
    output_trainset = np.zeros((4,input_trainset.shape[0],input_trainset.shape[1],input_trainset.shape[2]))
    output_trainset[0] = np.rot90(input_trainset,0,axes=(1,2))
    output_trainset[1] = np.rot90(input_trainset,1,axes=(1,2))
    output_trainset[2] = np.rot90(input_trainset,2,axes=(1,2))
    output_trainset[3] = np.rot90(input_trainset,3,axes=(1,2))
    return output_trainset

def parallel_diffuse_emission(path,i,j,k):
    background=np.load(path+"/background.npy")

    return background[i-64:i+64,j-64:j+64,k-64:k+64].astype(np.float32)

def generate_data():
    x1, y1, z1 = rr(64, 200), rr(800, 1000), rr(64, 4900)
    x2, y2, z2 = rr(64, 200), rr(64, 1025-64), rr(64, 1025-64)

    DE = parallel_diffuse_emission(input_path,x1, y1, z1)
    PS = point_sources(input_path, x2, y2, z2)

    return Augmentation(DE), Augmentation(DE + PS)

#if __name__ == "__main__":
#    pool = Pool(processes=40)  # Adjust the number of processes as needed
Train_DE=np.zeros((10240,128,128,128));Train_DEPS=np.zeros((10240,128,128,128))
for i in range(2560):
    Train_DE[4*i:4*i+4], Train_DEPS[4*i:4*i+4] = generate_data()

#    pool.close()
#    pool.join()

#    Train_DE = np.vstack(Train_DE)
#    Train_DEPS = np.vstack(Train_DEPS)

print("saving training set")
np.save(save_path + '/train_x', np.transpose(Train_DEPS,(0,2,3,1)).astype(np.float32))
print("DE+PS done")
np.save(save_path + '/train_y', np.transpose(Train_DE,(0,2,3,1)).astype(np.float32))
print("DE Done")

