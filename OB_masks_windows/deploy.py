import sys
import nibabel as nib
import numpy as np
import os
import gzip
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import gradio as gr
from zipfile import ZipFile

os.makedirs("output", exist_ok=True)
os.makedirs("unzip", exist_ok=True)

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
    snp = np.floor(snp)
    snp /= np.max(snp)  
    for img in snp :
        image_list.append(img)
    image_list = np.array(image_list)
    image_list = image_list.reshape(image_list.shape[0],image_list.shape[1],image_list.shape[2],1)
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

BB_IMAGE_HEIGHT = 32 
BB_IMAGE_WIDTH = 32   

def process_centers(model_r, scan):
    n =scan.shape[0]
    scan_w = scan.shape[1]
    scan_h = scan.shape[2]
    xp_sum=yp_sum=slicep_sum = 0
    for i in range(n):
        im = scan[i]
        im2 = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
        p=model_r.predict(im2)
        xp,yp = p[0]
        xp_sum+=xp
        yp_sum+=yp
    xp_med = xp_sum//n
    yp_med = yp_sum//n
    xi,xf = calc_i_f(scan_w,xp_med,BB_IMAGE_WIDTH)
    yi,yf = calc_i_f(scan_h,yp_med,BB_IMAGE_HEIGHT) 
    return  xi,xf,yi,yf 
    

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


def save_mask(ficheiro_scan, masks, file_name):
    mask_affine, mask_header=get_affine(ficheiro_scan)
    my_print(masks.shape)
    masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2])
    mask = np.flip(masks).T
    my_print(mask.shape)
    seg_mask = nib.Nifti1Image(mask, mask_affine, mask_header)
    nib.save(seg_mask, "./output/" + file_name + "_mask.nii.gz")
    return "./output/" + file_name + "_mask.nii.gz"

def unzip(file_path):
    file_name = file_path.rsplit('\\',1)[1]
    path = file_path.rsplit('\\',1)[0]
    with ZipFile(file_path, 'r') as zipObj:
        zipObj.extractall("./unzip")
    return file_name[:-26], path

def make_prediction(fich_scan_nii):

    file_name = fich_scan_nii.name[31:-26]
    file_path = fich_scan_nii.name
  
    file_name, path = unzip(file_path)
    
    file_path_unzip =  "./unzip/input/" + file_name + ".nii.gz"

    np_original_all_scan = generate_np_for_resizing(file_path_unzip)

    scan_BB, coord_BB = generate_BB_scan(model_c,model_r,np_original_all_scan)

    shape_scan = np_original_all_scan.shape

    masks = generate_mask(model_s,scan_BB, shape_scan, coord_BB)  

    output = save_mask(file_path_unzip, masks, file_name)

    return output



if __name__ == '__main__':
    print("start process")
    # load the models
    with tf.device('/cpu:0'):
        #model to classify images with bulb or without bulb
        model_c = load_model("modelo_CNN_treinado_v4.h5") 
        #model to calculate centers of images with bulb                  
        model_r = load_model("modelo_CNN_R_treinado_v4_174x190.h5")
        #segmentation model
        model_s = load_model("modelo_FCN_segmentation_32x32.h5" , compile=False)
        print("models lodaded successfully!")

    file_input = gr.inputs.File(file_count="single", type="file", label="File Input")
    file_output = gr.outputs.File(label="File Output")
    
    title = "Generate OB masks"
    description = "Description"

    gr.Interface(fn=make_prediction,
                inputs=file_input,
                outputs=file_output,
                title=title,
                description=description,
                server_port=8001, 
                server_name="0.0.0.0").launch(auth=('marcelo', '1234'), share=False)
    
    print("end process")