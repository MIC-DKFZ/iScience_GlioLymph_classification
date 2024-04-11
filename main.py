#    Copyright 2024 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0 DEED);
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://creativecommons.org/licenses/by-nc/4.0/deed.en
#


import sys
import numpy as np
import torch
import monai.transforms as T
import argparse
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import torch.nn as nn
from Dataset import UKHD_Dataset
import random
from torch.utils.data import DataLoader
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True


def predict_cases(model,loader,device):
    outTransform = nn.Sigmoid()
    labels = []
    predictions = []
    model=model.to(device)
    with torch.no_grad():
        for i in loader:
            images=i[0]
            label=i[1]
            images=images.to(device)
            images=images.type(torch.float32)
            result=model(images)
            result=outTransform(result)
            labels.extend([label.item()])
            predictions.extend([result.detach().cpu().numpy()])
        testPred=np.array(predictions)
        testLab=np.array(labels)
        score=roc_auc_score(testLab,testPred[:,0,0])
        fpr, tpr, threshold = roc_curve(testLab, testPred[:, 0, 0], pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr,label=f'Class (AUC = {score:.2f})')
        plt.legend(loc='lower right')
        return score

def parse_args(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument("-b","--batch_size",type=int,help="batch size",default=1)
    parser.add_argument("-p","--path",type=str,help="Path to data location",default="./")
    parser.add_argument("-s","--seed",type=int,help="Which Seed",default=42)
    parser.add_argument("-m","--model",type=str,help="Path to trained model",default="./densenet169_batch_14.pth")
    parser.add_argument("-d","--device",type=str,help="Device to use Cuda or CPU",default="cuda")
    args=parser.parse_args(argv)
    return args



def main(args):
    seed_everything(args.seed)
    model=torch.load(args.model)
    model.eval()
    trans_img = [T.ToTensor(),T.NormalizeIntensity()]
    transform=T.Compose(trans_img)
    trans_mask = [T.ToTensor()]
    transform_mask = T.Compose(trans_mask)
    device = torch.device(args.device)
    DatasetClass=UKHD_Dataset(args.path,'test',[0,1,2,3], transform_image=transform, transform_mask=transform_mask)
    loader=DataLoader(DatasetClass,batch_size=args.batch_size,shuffle=False)
    score=predict_cases(model,loader,device)
    print(f"Accuracy Score on the provided data: {score}")


if __name__=="__main__":
    args=parse_args(sys.argv[1:])
    main(args)





