#    Copyright 2024 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0 DEED);
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://creativecommons.org/licenses/by-nc/4.0/deed.en
#

# 0 = Glioblastom und optimal_thresh=0.5


import sys
import os
import numpy as np
import torch
import monai.transforms as T
import argparse
import torch.nn as nn
import random
import nibabel as nib


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True


def predict_cases(model,images,device):
    outTransform = nn.Sigmoid()
    model=model.to(device)
    with torch.no_grad():
        images=images.to(device)
        images=images.type(torch.float32)
        images=images.unsqueeze(0)
        result=model(images)
        result=outTransform(result)
        if result <= 0.5:
            result = 0
            prediction='Glioblastom'
        else:
            result = 1
            prediction='Lymphom'
        return prediction



def parse_args(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument("-b","--batch_size",type=int,help="batch size",default=1)
    parser.add_argument("-p","--path",type=str,help="Path to data location",default="/inputdata")
    parser.add_argument("-s","--seed",type=int,help="Which Seed",default=42)
    parser.add_argument("-m","--model",type=str,help="Path to trained model",default="./densenet169_batch_14.pth")
    parser.add_argument("-d","--device",type=str,help="Device to use Cuda or CPU",default="cpu")
    args=parser.parse_args(argv)
    return args



def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    model=torch.load(args.model,map_location=torch.device('cpu'))
    model.eval()
    trans_img = [T.ToTensor(),T.NormalizeIntensity()]
    transform=T.Compose(trans_img)
    trans_mask = [T.ToTensor()]
    transform_mask = T.Compose(trans_mask)
    basePath=args.path
    _,Subjects,_=next(os.walk(basePath))
    for subject in Subjects:
        _,_,images=next(os.walk(os.path.join(basePath,subject)))
        images=[image for image in images if 'seg' not in image]
        image_name = images[0][:-len('_0000.nii.gz')]
        print(image_name)
        image_list= [nib.load(os.path.join(basePath,subject,image_name+'_000'+str(f)+'.nii.gz')).get_fdata() for f in range(0,4)]
        mask=nib.load(os.path.join(basePath,subject,image_name+'_seg.nii.gz')).get_fdata()
        image_list = [transform(image) for image in image_list]
        mask = transform_mask(mask)
        image_list.append(mask)
        image_list = torch.stack(image_list)
        image_list = image_list.squeeze(dim = 1)


        prediction=predict_cases(model,image_list,device)
        print(f"Predicted Label: {prediction}")



if __name__=="__main__":
    args=parse_args(sys.argv[1:])
    main(args)








