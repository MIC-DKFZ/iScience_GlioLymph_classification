import nibabel as nib
import os
import torch
from torch.utils.data import Dataset
import monai



class UKHD_Dataset(Dataset):
    def __init__(self, path, mode, modalities, transform_image=None, transform_mask = None):
        """
        :type path: str
        :param path:
        :param mode:
        :param modalities:
        :param transform_image:
        :param transform_mask:
        This class is used to load the data from the dataset. It is used in the main.py file.
        The Dataset is loaded from the path, which is given in the main.py file. The mode is either 'train' or 'test'.
        The modalities are the different modalities of the MRI images. The transform_image and transform_mask are used to
        transform the images and the masks. The class_names are the different classes of the dataset.
        The Dataset expects the data to be in the following format:
        path/
                ├── Glioblastom
                        ├── img_caseG1_0000.nii.gz (FLAIR)
                        ├── img_caseG1_0001.nii.gz (T1)
                        ├── img_caseG1_0002.nii.gz (T1ce)
                        ├── img_caseG1_0003.nii.gz (T2)
                        ├── img_caseG1_seg.nii.gz (segmentation)
                        ├── img_caseG2_0000.nii.gz (FLAIR)
                        ├── img_caseG2_0001.nii.gz (T1)
                        ├── img_caseG2_0002.nii.gz (T1ce)
                        ├── img_caseG2_0003.nii.gz (T2)
                        ├── img_caseG2_seg.nii.gz (segmentation)
                        └── ...
                └── Lymphom
                        ├── img_caseL1_0000.nii.gz (FLAIR)
                        ├── img_caseL1_0001.nii.gz (T1)
                        ├── img_caseL1_0002.nii.gz (T1ce)
                        ├── img_caseL1_0003.nii.gz (T2)
                        ├── img_caseL1_seg.nii.gz (segmentation)
                        ├── img_caseL2_0000.nii.gz (FLAIR)
                        ├── img_caseL2_0001.nii.gz (T1)
                        ├── img_caseL2_0002.nii.gz (T1ce)
                        ├── img_caseL2_0003.nii.gz (T2)
                        ├── img_caseL2_seg.nii.gz (segmentation)
                        └── ...

        """
        self.path = path
        self.mode = mode
        self.modalities = modalities # Flair = 0000, T1 = 0001, T1ce = 0002, T2 = 0003 (Order is important)
        self.transform_image = transform_image #transformations to dataset
        self.transform_mask = transform_mask #transformations to mask
        self.class_names = ['Glioblastom', 'Lymphom']
        self.image_list = []
        self.mask_list = []
        self.label_list = []

        for class_name in self.class_names:
            class_path = os.path.join(self.path, self.mode, class_name)
            for image_name in sorted(os.listdir(class_path)):
                if image_name.endswith('_seg.nii.gz'):
                    self.mask_list.append(os.path.join(class_path, f"{image_name}"))
                else:
                    if '0000' in image_name:
                        image_name = image_name[:-len('_0000.nii.gz')]
                        image_path = [os.path.join(class_path, f"{image_name}_000{modality}.nii.gz") for modality in self.modalities]
                        self.image_list.append(image_path)
                        self.label_list.append(self.class_names.index(class_name))


    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        images = [nib.load(image_path).get_fdata() for image_path in self.image_list[idx]]
        mask = nib.load(self.mask_list[idx]).get_fdata()
        label = self.label_list[idx]
        if self.transform_image:
            images = [self.transform_image(image) for image in images]
        else:
            images = [monai.transforms.ToTensor()(image) for image in images]
        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = monai.transforms.ToTensor()(mask)
        images.append(mask)
        img = torch.stack(images)
        img = img.squeeze(dim = 1)
        return img, label