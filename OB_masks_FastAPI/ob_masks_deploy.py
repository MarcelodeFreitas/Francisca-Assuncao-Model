import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from zipfile import ZipFile

import os
import logging

""" logging.basicConfig(filename=(os.path.basename(__file__)[0:-3]+'.log'), level=logging.ERROR) """

model_c = ""
model_r = ""
model_s = ""  

def generate_np_for_resizing(ficheiro_scan):
    try:
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
        print("generate_np_for_resizing")
        return image_list
    except: 
        logging.exception("generate_np_for_resizing: ")

def calc_i_f(n,med,nbb):
    try:
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
    except: 
        logging.exception("calc_i_f: ")

BB_IMAGE_HEIGHT = 32 
BB_IMAGE_WIDTH = 32   

def process_centers(model_r, scan):
    try:
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
    except: 
        logging.exception("process_centers: ")
    
def process_bulnobulb(model_c, scan):
    try:
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
    except: 
        logging.exception("process_bulnobulb: ")
    
def generate_BB_scan(model_c, model_r, scan):
    try:
        si,sf = process_bulnobulb(model_c, scan)
        if si==sf :
            return None
        else:
            xi,xf,yi,yf = process_centers(model_r, scan[si:sf,:,:])  
            print("generate_BB_scan")
        return scan[si:sf,xi:xf,yi:yf], (si,sf,xi,xf,yi,yf)
    except: 
        logging.exception("generate_BB_scan: ")

def get_affine(ficheiro_scan):
    try:
        scan = nib.load(ficheiro_scan)
        mask_affine=scan.affine
        mask_header=scan.header
        return mask_affine,mask_header
    except: 
        logging.exception("get_affine: ")

def predict_pos_processing(mask,threshold):
    try:
        return np.where(mask > threshold, 1., 0.)
    except: 
        logging.exception("predict_pos_processing: ")


def generate_mask(model_s,scan_BB, shape_scan,coord_BB):
    try:
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
    except: 
        logging.exception("generate_mask: ")

def save_mask(ficheiro_scan, masks, file_name, ouput_directory):
    try:
        mask_affine, mask_header=get_affine(ficheiro_scan)
        masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2])
        mask = np.flip(masks).T
        seg_mask = nib.Nifti1Image(mask, mask_affine, mask_header)
        nib.save(seg_mask, ouput_directory + file_name)
        return ouput_directory + file_name
    except: 
        logging.exception("save_mask: ")

def unzip(file_path):
    try:
        file_name = file_path.rsplit('/',1)[1]
        path = file_path.rsplit('/',1)[0]
        with ZipFile(file_path, 'r') as zipObj:
            zipObj.extractall("./unzip")
        return file_name[:-26], path
    except: 
        logging.exception("unzip: ")

def make_prediction(fich_scan_nii):
    try: 

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
    except: 
        logging.exception("make_prediction: ")

def load_models(modelpaths, input_file_path, output_file_name, output_directory_path):
    try:
        model_c = None
        model_r = None
        model_s = None
        try:
            with tf.device('/cpu:0'):
                for i in modelpaths:
                    name = i["name"]
                    path = i["path"]
                    if (name == "modelo_CNN_R_treinado_v4_174x190.h5"):
                        model_c = load_model(path)
                        print("model_c: ", model_c)
                    if (name == "modelo_CNN_treinado_v4.h5"):
                        model_r = load_model(path)
                        print("model_r: ", model_r)
                    if (name == "modelo_FCN_segmentation_32x32.h5"):
                        model_s = load_model(path, compile=False)
                        print("model_s: ", model_s)
        except: 
            return False

        np_original_all_scan = generate_np_for_resizing(input_file_path)
        scan_BB, coord_BB = generate_BB_scan(model_c,model_r,np_original_all_scan)
        shape_scan = np_original_all_scan.shape
        masks = generate_mask(model_s,scan_BB, shape_scan, coord_BB)  
        save_mask(output_directory_path, masks, output_file_name)
    except:
        logging.exception("load_models: ")
    
def run(input_file_path, output_file_name, output_directory_path):
    try:
        np_original_all_scan = generate_np_for_resizing(input_file_path)
        scan_BB, coord_BB = generate_BB_scan(model_c,model_r,np_original_all_scan)
        shape_scan = np_original_all_scan.shape
        masks = generate_mask(model_s,scan_BB, shape_scan, coord_BB)  
        save_mask(output_directory_path, masks, output_file_name)
    except:
        logging.exception("run: ")