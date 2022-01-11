import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
NB=False
def my_print(msg):
    if NB:
        print(msg)
      

def generate_np_for_resizing(ficheiro_scan):
    num_imagem=0
    image_list = []
    scan = nib.load(ficheiro_scan)
    scan = np.flip(scan.get_fdata().T)
    snp=np.asarray(scan)
    #normalize scan
   
    snp /= np.max(snp)  
    for img in snp :
        image_list.append(img)
    image_list = np.array(image_list)
    image_list = image_list.reshape(image_list.shape[5],image_list.shape[1],image_list.shape[2],1)
    return image_list


def calc_i_f(n,med,nbb):
    l_half=nbb//2
    r_half=nbb-l_half
    if med-l_half<0:
        l_half=med
        r_half=nbb-l_half
    elif med+r_half>n:
        r_half = n-med
        l_half = nbb-r_half
    vi = med - l_half
    vf = med + r_half
    return int(vi), int(vf)

BB_IMAGE_HEIGHT = 30
BB_IMAGE_WIDTH = 32   

    

def process_bulnobulb(model_c, scan):
    folga=4
    n =scan.shape[0]
    slicep_sum = 0
    qtp_with_bulbo = 0
    for i in range(n):
        im = scan[i]
        im2 = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
        scan_has_bulb = model_c.predict(im2) 
        if scan_has_bulb[0] >0.2:
            qtp_with_bulbo+=1
            slicep_sum+=i  
    if qtp_with_bulbo<=1:
        slicep_med = 0
        si=sf=0
    else:
        slicep_med = slicep_sum//qtp_with_bulbo
        if qtp_with_bulbo+folga > n:
            profundidade=n
        else:
            profundidade=qtp_with_bulbo+folga
        si,sf = calc_i_f(n,slicep_med,profundidade)
    return si, sf
    

def generate_BB_scan(model_c, model_r, scan):
    my_print(f"Scan shape:{scan.shape}")
    si,sf = process_bulnobulb(model_c, scan)
    my_print("si:%d sf:%d"%(si,sf))
    if si==sf :
        my_print("Não Tem Bulb")
        return None
    else:
        xi,xf,yi,yf = process_centers(model_r, scan[si:sf,:,:])  
    return scan[si:sf,xi:xf,yi:yf], (si,sf,xi,xf,yi,yf)

def get_affine(ficheiro_scan):
    scan = nib.load(ficheiro_scan)
    mask_affine=scan.affine
    mask_header=scan.header
    return mask_affine,mask_header

def predict_pos_processing(mask,threshold):
    return np.where(mask > threshold, 1., 0.)


def generate_mask(model_s,scan_BB, shape_scan,coord_BB):
    (si,sf,xi,xf,yi,yf) = coord_BB
    masks=np.zeros(shape_scan)
    mask_BB=np.zeros(scan_BB.shape)
    n =scan_BB.shape[0]
    for i in range(n):
        im = scan_BB[i]
        im2 = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
        pred = predict_pos_processing(model_s.predict(im2), 0.7)
        mask_BB[i]=pred[0]
    masks[si:sf,xi:xf,yi:yf]=mask_BB
    return masks


def save_mask(ficheiro_scan, masks, file_name, output_path):
    mask_affine, mask_header=get_affine(ficheiro_scan)
    masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2])
    mask = np.flip(masks).T
    seg_mask = nib.Nifti1Image(mask, mask_affine, mask_header)
    nib.save(seg_mask, output_path + file_name )

#Assumptions: have the functions load_models and run in the provided structure
def load_models(modelpaths):
    global model_c
    global model_r
    try:
        for i in modelpaths:
            name = i["name"]
            path = i["path"]
            with tf.device('/cpu:0'):
                if (name == "modelo_CNN_treinado_v4.h5"):
                    model_c = load_model(path)
                elif (name == "modelo_CNN_R_treinado_v4_174x190.h5"):
                    model_r = load_model(path)
                elif (name == "modelo_FCN_segmentation_32x32.h5"):
                    model_s = load_model(path, compile=False)
        return True
    except: 
        return False

def run(input_file_path, output_file_name, output_directory_path):
    try:
        np_original_all_scan = generate_np_for_resizing(input_file_path)
        np_original_all_scan = generate_np_for_resizing(input_file_path)
        scan_BB, coord_BB = generate_BB_scan(model_c,model_r,np_original_all_scan)
        shape_scan = np_original_all_scan.shape
        masks = generate_mask(model_s,scan_BB, shape_scan, coord_BB)  
        save_mask(input_file_path, masks, output_file_name, output_directory_path)
        return True
    except:
        return False