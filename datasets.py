import os
import numpy as np
import cv2
import h5py
import json

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter


class Gaze360(Dataset):
    def __init__(self, path, root, transform, angle, binwidth, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        if train==False:
          angle=90
        self.binwidth=binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)        
        
        # Bin values
        bins_yaw = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_yaw = np.digitize(yaw, bins_yaw) - 1
        bins_pitch = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_pitch = np.digitize(pitch, bins_pitch) - 1

        labels = np.array([binned_pitch,binned_yaw])
        cont_labels = torch.FloatTensor([pitch, yaw])
        #print('real label:',labels,"\n")
        

        return img, labels, cont_labels, name

class Mpiigaze(Dataset): 
  def __init__(self, pathorg, root, transform, train, angle,fold=0):
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      path.pop(fold)
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 42 and abs((label[1]*180/np.pi)) <= 42:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    pitch = label[0]* 180 / np.pi    # 弧度转换成角度
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    bins = np.array(range(-42, 42,2))
    binned_pose = np.digitize([pitch, yaw], bins) - 1    #分组

    labels = binned_pose
    cont_labels = torch.FloatTensor([pitch, yaw])    #2个


    return img, labels, cont_labels, name

class HDFDataset(Dataset):
    """Dataset from HDF5 archives formed of 'groups' of specific persons."""

    def __init__(self, hdf_file_path,
                 prefixes=None,
                 is_bgr=False,
                 pick_at_least_per_person=None,
                 num_labeled_samples=None,
                 transform=None
                 ):
        assert os.path.isfile(hdf_file_path)
        self.hdf_path = hdf_file_path
        self.hdf = None
        self.is_bgr = is_bgr
        self.transform = transform

        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            if prefixes is None:
                self.prefixes = hdf_keys
            else:
                self.prefixes = [k for k in prefixes if k in h5f]
            if pick_at_least_per_person is not None:
                self.prefixes = [k for k in self.prefixes if k in h5f and len(next(iter(h5f[k].values()))) >=
                            pick_at_least_per_person]
            self.index_to_query = sum([[(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                                       for prefix in self.prefixes], [])
            if num_labeled_samples is not None:
                # randomly pick labeled samples for semi-supervised training
                ra = list(range(len(self.index_to_query)))
                # random.seed(0)
                # random.shuffle(ra)
                # Make sure that the ordering is the same
                # assert ra[:3] == [744240, 1006758, 1308368]
                ra = ra[:num_labeled_samples]
                list.sort(ra)
                self.index_to_query = [self.index_to_query[i] for i in ra]

            # calculate kernel density of gaze and head pose, for generating new redirected samples

    def __len__(self):
        return len(self.index_to_query)

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def preprocess_image(self, image):
        if self.is_bgr:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        image = Image.fromarray(image)
        
        # image = 2.0 * image / 255.0 - 1
        image = self.transform(image)
        # image = np.transpose(image, [2, 0, 1])  # Colour image
        return image


    def __getitem__(self, idx):
        if self.hdf is None:  # Need to lazy-open this to avoid read error
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Pick entry a and b from same person
        key_a, idx_a = self.index_to_query[idx]
        group_a = self.hdf[key_a]

        def retrieve(group, index):
            eyes = self.preprocess_image(group['pixels'][index, :])
            g = group['labels'][index, :2]
            g = torch.from_numpy(g).type(torch.FloatTensor)

            return eyes, g
        # Grab 1st (input) entry
        img, g_a = retrieve(group_a, idx_a)

        #弧度转化为角度
        yaw = g_a[0]* 180 / np.pi
        pitch = g_a[1]* 180 / np.pi
        # Bin values
        bins = np.array(range(-42, 42, 2))
        binned_pose = np.digitize([pitch, yaw], bins) - 1
        cont_labels = torch.FloatTensor([pitch, yaw])

        # return self.preprocess_entry(entry)
        return img, binned_pose, cont_labels

class RTGene(Dataset):
    def __init__(self,root,path,transform,angle):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.lines = []

        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[6]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        line = line.strip().split(" ")

        face = line[0]
        name = line[3]
        gaze2d = line[6]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi    # 弧度转换成角度
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        if self.transform:
            img = self.transform(img)        

        # Bin values,这个试一下 2度 分类
        bins = np.array(range(-38, 38,2))
        binned_pose = np.digitize([pitch, yaw], bins) - 1    #分组

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])    #2个

        return img, labels, cont_labels, name

class EyeDiap(Dataset):
    def __init__(self,root,path,transform,angle):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.lines = []

        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[6]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        line = line.strip().split(" ")

        face = line[0]
        name = line[3]
        gaze2d = line[6]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi    # 弧度转换成角度
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        if self.transform:
            img = self.transform(img)        

        # Bin values,这个试一下 2度 分类
        bins = np.array(range(-42, 42,2))
        binned_pose = np.digitize([pitch, yaw], bins) - 1    #分组

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])    #2个

        return img, labels, cont_labels, name

class EyeDiap_leaveOne(Dataset):
    def __init__(self, pathorg, root, transform, train, angle, fold=0):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.lines = []
        path = pathorg.copy()
        if train == True:
            path.pop(fold)
        else:
            path = path[fold]
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    lines = f.readlines()
                    lines.pop(0)
                    self.orig_list_len += len(lines)
                    for line in lines:
                        gaze2d = line.strip().split(" ")[7]
                        label = np.array(gaze2d.split(",")).astype("float")
                        if abs((label[0] * 180 / np.pi)) <= angle and abs((label[1] * 180 / np.pi)) <= angle:
                            self.lines.append(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0] * 180 / np.pi)) <= angle and abs((label[1] * 180 / np.pi)) <= angle:
                        self.lines.append(line)

        print(
            "{} items removed from dataset that have an angle > {}".format(self.orig_list_len - len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        name = line[3]
        gaze2d = line[7]
        head2d = line[8]
        lefteye = line[1]
        righteye = line[2]
        face = line[0]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0] * 180 / np.pi  # 弧度转换成角度
        yaw = label[1] * 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)

            # Bin values
        bins = np.array(range(-30, 30, 2))
        binned_pose = np.digitize([pitch, yaw], bins) - 1  # 分组

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])  # 2个

        return img, labels, cont_labels, name


# 多尺度分类
class Mpii_multi(Dataset): 
  def __init__(self, pathorg, root, transform, train, angle,fold=0):
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      path.pop(fold)
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 42 and abs((label[1]*180/np.pi)) <= 42:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    pitch = label[0]* 180 / np.pi    # 弧度转换成角度
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    bins = np.array(range(-42, 42,1))
    binned_pose = np.digitize([pitch, yaw], bins) - 1    #分组

    bins_0 = np.array(range(-42, 42,3))
    binned_pose_0 = torch.LongTensor(np.digitize([pitch, yaw], bins_0) - 1)
    
    bins_1 = np.array(range(-42, 42,7))
    binned_pose_1 = torch.LongTensor(np.digitize([pitch, yaw], bins_1) - 1)
    
    bins_2 = np.array(range(-42, 42,21))
    binned_pose_2 = torch.LongTensor(np.digitize([pitch, yaw], bins_2) - 1)
    
    bins_3 = np.array(range(-42, 42,42))
    binned_pose_3 = torch.LongTensor(np.digitize([pitch, yaw], bins_3) - 1)




    labels = binned_pose
    labels_0 = binned_pose_0
    labels_1 = binned_pose_1
    labels_2 = binned_pose_2
    labels_3 = binned_pose_3
    cont_labels = torch.FloatTensor([pitch, yaw])    #2个


    return img, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name

class Gaze360_multi(Dataset):
    def __init__(self, path, root, transform, angle, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        if train==False:
          angle=90
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)        
        
        # Bin values
        bins = np.array(range(-1*self.angle, self.angle, 2))
        binned_pose = np.digitize([pitch, yaw], bins) - 1    #分组

        bins_0 = np.array(range(-1*self.angle, self.angle,3))
        binned_pose_0 = torch.LongTensor(np.digitize([pitch, yaw], bins_0) - 1)
        
        bins_1 = np.array(range(-1*self.angle, self.angle,10))
        binned_pose_1 = torch.LongTensor(np.digitize([pitch, yaw], bins_1) - 1)
        
        bins_2 = np.array(range(-1*self.angle, self.angle,30))
        binned_pose_2 = torch.LongTensor(np.digitize([pitch, yaw], bins_2) - 1)
        
        bins_3 = np.array(range(-1*self.angle, self.angle,90))
        binned_pose_3 = torch.LongTensor(np.digitize([pitch, yaw], bins_3) - 1)


        labels = binned_pose
        labels_0 = binned_pose_0
        labels_1 = binned_pose_1
        labels_2 = binned_pose_2
        labels_3 = binned_pose_3
        cont_labels = torch.FloatTensor([pitch, yaw])    #2个

        return img, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name

class EyeDiap_multi(Dataset):
    def __init__(self,root,path,transform,angle):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.lines = []

        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[6]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        line = line.strip().split(" ")

        face = line[0]
        name = line[3]
        gaze2d = line[6]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi    # 弧度转换成角度
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        if self.transform:
            img = self.transform(img)        


        # Bin values
        bins = np.array(range(-42, 42,1))
        binned_pose = np.digitize([pitch, yaw], bins) - 1    #分组

        bins_0 = np.array(range(-42, 42,3))
        binned_pose_0 = torch.LongTensor(np.digitize([pitch, yaw], bins_0) - 1)
        
        bins_1 = np.array(range(-42, 42,7))
        binned_pose_1 = torch.LongTensor(np.digitize([pitch, yaw], bins_1) - 1)
        
        bins_2 = np.array(range(-42, 42,21))
        binned_pose_2 = torch.LongTensor(np.digitize([pitch, yaw], bins_2) - 1)
        
        bins_3 = np.array(range(-42, 42,42))
        binned_pose_3 = torch.LongTensor(np.digitize([pitch, yaw], bins_3) - 1)




        labels = binned_pose
        labels_0 = binned_pose_0
        labels_1 = binned_pose_1
        labels_2 = binned_pose_2
        labels_3 = binned_pose_3
        cont_labels = torch.FloatTensor([pitch, yaw])    #2个


        return img, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name


if __name__ == '__main__':
    with open('./gazecapture_split.json', 'r') as f:
        all_gc_prefixes = json.load(f)


    train_dataset = HDFDataset(hdf_file_path="E:\EyeTracking\Code\datasets\GazeCapture_128.h5", prefixes=all_gc_prefixes['train'],)
    print('Loading data.')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=False)

