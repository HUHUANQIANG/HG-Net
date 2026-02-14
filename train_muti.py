import os
import argparse
from pyexpat import model
import time
import numpy as np

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim


import datasets
from model import L2CS, R_WGENet, Multinet_mpii, Multinet_360, Multinet_eye
from utils import select_device, gazeto3d, angular
from efficientnet_pytorch import EfficientNet

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/train.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)
    #RTgene
    parser.add_argument(
        '--gazeRTgene_dir', dest='gazeRTgene_dir', help='Directory path for gaze images.',
        default='datasets/RT-GENE', type=str)
    parser.add_argument(
        '--gazeRTgenelabel_dir', dest='gazeRTgenelabel_dir', help='Directory path for gaze labels.',
        default='datasets/RT-GENE/Label/train/1/train_1&2.label', type=str)
    #EyeDiap
    parser.add_argument(
        '--gazeEyeDiap_dir', dest='gazeEyeDiap_dir', help='Directory path for gaze images.',
        default='datasets/EYEDIAP/Image', type=str)
    parser.add_argument(
        '--gazeEyeDiaplabel_dir', dest='gazeEyeDiaplabel_dir', help='Directory path for gaze labels.',
        default='datasets/EYEDIAP/ClusterLabel/3/train301.label', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='1', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=40, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=30, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=2, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.001, type=float)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
                
def load_filtered_state_dict(model, path):
    # By user apaszke from discuss.pytorch.org
    
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)



def getArch_weights(arch, bins):
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'models/resnet50-19c8e357.pth'

    return model, pre_url

# 宽角度损失函数,range由自己的数据集决定
def wrapLoss(true,pred,range):
    return torch.mean(
        torch.minimum(
            abs(pred - true)**2,(range - abs(pred - true))**2)
            )


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    #gpu = torch.device("cpu")
    
    data_set=args.dataset
    alpha = args.alpha
    output=args.output
    
    
    transformations = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    
    
    if data_set=="gaze360":

        # Mutil
        model = Multinet_360(yaw_num_bins=180, pitch_num_bins=180)        
        model.cuda(gpu)
        dataset=datasets.Gaze360_multi(args.gaze360label_dir, args.gaze360image_dir, transformations, 90)


        print('Loading data.')
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=False)
        torch.backends.cudnn.benchmark = True

        val_path = "datasets/Gaze360/Label/val.label"
        val_dataset=datasets.Gaze360_multi(val_path,args.gaze360image_dir, transformations, 90, train=False)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=4,
            pin_memory=False)

        summary_name = '{}'.format('Mutil-gaze360')
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)


        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(180)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)      

        # Optimizer gaze
        optimizer_gaze = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gaze, step_size=28000, gamma=0.1, last_epoch=-1)
       

        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)

        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
            print("第%d个epoch的学习率：%f" % (epoch, optimizer_gaze.param_groups[0]['lr']))
            
            for i, (images_gaze, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)

                # Binned labels
                label_pitch = Variable(labels[:, 0]).cuda(gpu)
                label_yaw = Variable(labels[:, 1]).cuda(gpu)

                label_yaw_0 = Variable(labels_0[:,0]).cuda(gpu)
                label_pitch_0 = Variable(labels_0[:,1]).cuda(gpu)

                label_yaw_1 = Variable(labels_1[:,0]).cuda(gpu)
                label_pitch_1 = Variable(labels_1[:,1]).cuda(gpu)
                
                label_yaw_2 = Variable(labels_2[:,0]).cuda(gpu)
                label_pitch_2 = Variable(labels_2[:,1]).cuda(gpu)
                
                label_yaw_3 = Variable(labels_3[:,0]).cuda(gpu)
                label_pitch_3 = Variable(labels_3[:,1]).cuda(gpu)


                # Continuous labels
                label_pitch_cont = Variable(cont_labels[:, 0]).cuda(gpu)
                label_yaw_cont = Variable(cont_labels[:, 1]).cuda(gpu)
                pre_pitch, pre_pitch_0, pre_pitch_1, pre_pitch_2, pre_pitch_3, pre_yaw, pre_yaw_0, pre_yaw_1, pre_yaw_2, pre_yaw_3 = model(images_gaze)


                # Cross entropy loss
                loss_pitch,loss_pitch_0,loss_pitch_1,loss_pitch_2, loss_pitch_3 = criterion(pre_pitch, label_pitch),criterion(pre_pitch_0, label_pitch_0),criterion(pre_pitch_1, label_pitch_1),criterion(pre_pitch_2, label_pitch_2),criterion(pre_pitch_3, label_pitch_3)
                loss_yaw,loss_yaw_0,loss_yaw_1,loss_yaw_2, loss_yaw_3 = criterion(pre_yaw, label_yaw),criterion(pre_yaw_0, label_yaw_0),criterion(pre_yaw_1, label_yaw_1),criterion(pre_yaw_2, label_yaw_2),criterion(pre_yaw_2, label_yaw_2)

                # MSE loss
                pitch_predicted = softmax(pre_pitch)
                yaw_predicted = softmax(pre_yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 1 - 90
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 1 - 90

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont)  


                # one batch Total loss
                loss_pitch_gaze = alpha * loss_reg_pitch + 7*loss_pitch + 4*loss_pitch_0 + 2*loss_pitch_1 + 0.8*loss_pitch_2 + 0.8*loss_pitch_3
                loss_yaw_gaze = alpha * loss_reg_yaw + 7*loss_yaw + 4*loss_yaw_0 + 2*loss_yaw_1 + 0.8*loss_yaw_2 + 0.8*loss_yaw_3
                

                # all batch loss
                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = \
                    [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()
                scheduler.step()

                iter_gaze += 1

                if (i+1) % 500 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )


            val_loss=[]
            epoch_list=[]
            avg_MAE=[]
            total = 0
            avg_error = .0

            with torch.no_grad():
                for j, (images, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name) in enumerate(val_loader):
                    images = Variable(images).cuda(gpu)
                    total += cont_labels.size(0)

                    #角度转化为弧度
                    label_pitch = cont_labels[:,0].float()*np.pi/180
                    label_yaw = cont_labels[:,1].float()*np.pi/180
                    
                    pre_pitch, pre_pitch_0, pre_pitch_1, pre_pitch_2, pre_pitch_3, pre_yaw, pre_yaw_0, pre_yaw_1, pre_yaw_2, pre_yaw_3 = model(images)
                    # Binned predictions
                    _, pitch_bpred = torch.max(pre_pitch.data, 1)
                    _, yaw_bpred = torch.max(pre_yaw.data, 1)
                    
        
                    # Continuous predictions
                    pitch_predicted = softmax(pre_pitch)
                    yaw_predicted = softmax(pre_yaw)
                    
                    # mapping from binned (0 to 28) to angels (-42 to 42)                
                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 1 - 90
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 1 - 90
                    
                    #角度转化为弧度
                    pitch_predicted = pitch_predicted*np.pi/180
                    yaw_predicted = yaw_predicted*np.pi/180

                    for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                        error = angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                        avg_error += error
                        val_loss.append(error)    
                    
                arr_var = np.var(val_loss)
                arr_std = np.std(val_loss)
                np_avg = np.mean(val_loss)
                print("val angular:",avg_error/ total)
                print("val numpy angular:",np_avg)
                print("len val_loss:",len(val_loss))
                print("val arr_var:",arr_var)
                print("val arr_std:",arr_std)
          
                torch.save(model.state_dict(),
                            output +'/'+
                            'epoch_' + str(epoch+1) + '.pkl')


   
    elif data_set=="mpiigaze":
        folder = os.listdir(args.gazeMpiilabel_dir)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]
        for fold in range(0,15):

            model = Multinet_mpii(yaw_num_bins=84, pitch_num_bins=84)  
            model.cuda(gpu)
            print('Loading data.')
            dataset=datasets.Mpii_multi(testlabelpathombined,args.gazeMpiimage_dir, transformations, True, angle=42, fold=fold)
            train_loader_gaze = DataLoader(
                dataset=dataset,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            torch.backends.cudnn.benchmark = True

            summary_name = '{}'.format('Muti-mpiigaze')
            if not os.path.exists(os.path.join(output+'/','fold' + str(fold))):
                os.makedirs(os.path.join(output+'/','fold' + str(fold)))

            criterion = nn.CrossEntropyLoss().cuda(gpu)
            reg_criterion = nn.MSELoss().cuda(gpu)
            softmax = nn.Softmax(dim=1).cuda(gpu)
            idx_tensor = [idx for idx in range(84)]
            idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

            # Optimizer gaze
            optimizer_gaze = optim.Adam(model.parameters(), lr=lr)
        
            configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}, fold={fold}--------------\n"
            print(configuration)
            
            
            for epoch in range(num_epochs):
                sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

                
                for i, (images_gaze, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name) in enumerate(train_loader_gaze):
                    images_gaze = Variable(images_gaze).cuda(gpu)

                    # Binned labels
                    label_pitch = Variable(labels[:, 0]).cuda(gpu)
                    label_yaw = Variable(labels[:, 1]).cuda(gpu)

                    label_yaw_0 = Variable(labels_0[:,0]).cuda(gpu)
                    label_pitch_0 = Variable(labels_0[:,1]).cuda(gpu)

                    label_yaw_1 = Variable(labels_1[:,0]).cuda(gpu)
                    label_pitch_1 = Variable(labels_1[:,1]).cuda(gpu)
                    
                    label_yaw_2 = Variable(labels_2[:,0]).cuda(gpu)
                    label_pitch_2 = Variable(labels_3[:,1]).cuda(gpu)
                    
                    label_yaw_3 = Variable(labels_3[:,0]).cuda(gpu)
                    label_pitch_3 = Variable(labels_3[:,1]).cuda(gpu)


                    # Continuous labels
                    label_pitch_cont = Variable(cont_labels[:, 0]).cuda(gpu)
                    label_yaw_cont = Variable(cont_labels[:, 1]).cuda(gpu)

                    pre_pitch, pre_pitch_0, pre_pitch_1, pre_pitch_2, pre_pitch_3, pre_yaw, pre_yaw_0, pre_yaw_1, pre_yaw_2, pre_yaw_3 = model(images_gaze)

                    # Cross entropy loss
                    loss_pitch,loss_pitch_0,loss_pitch_1,loss_pitch_2,loss_pitch_3 = criterion(pre_pitch, label_pitch),criterion(pre_pitch_0, label_pitch_0),criterion(pre_pitch_1, label_pitch_1),criterion(pre_pitch_2, label_pitch_2),criterion(pre_pitch_3, label_pitch_3)
                    loss_yaw,loss_yaw_0,loss_yaw_1,loss_yaw_2,loss_yaw_3 = criterion(pre_yaw, label_yaw),criterion(pre_yaw_0, label_yaw_0),criterion(pre_yaw_1, label_yaw_1),criterion(pre_yaw_2, label_yaw_2),criterion(pre_yaw_3, label_yaw_3)

                    # MSE loss
                    pitch_predicted = softmax(pre_pitch)
                    yaw_predicted = softmax(pre_yaw)

                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 1 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 1 - 42

                    loss_reg_pitch = reg_criterion(
                       pitch_predicted, label_pitch_cont)
                    loss_reg_yaw = reg_criterion(
                       yaw_predicted, label_yaw_cont)
                        
                    # loss_reg_pitch = wrapLoss(
                    #     pitch_predicted, label_pitch_cont_gaze,range = 84).cuda(gpu)
                    # loss_reg_yaw = wrapLoss(
                    #     yaw_predicted, label_yaw_cont_gaze,range = 84).cuda(gpu)

                    # one batch Total loss
                    loss_pitch_gaze = alpha * loss_reg_pitch + 7*loss_pitch + 4*loss_pitch_0 + 2*loss_pitch_1 + 0.8*loss_pitch_2 + 0.8*loss_pitch_3
                    loss_yaw_gaze = alpha * loss_reg_yaw + 7*loss_yaw + 4*loss_yaw_0 + 2*loss_yaw_1 + 0.8*loss_yaw_2 + 0.8*loss_yaw_3

                    # all batch loss
                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = \
                        [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                    optimizer_gaze.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss_seq, grad_seq)
                    optimizer_gaze.step()

                    iter_gaze += 1

                    if (i+1) % 500 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                            'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                                epoch+1,
                                num_epochs,
                                i+1,
                                len(dataset)//batch_size,
                                sum_loss_pitch_gaze/iter_gaze,
                                sum_loss_yaw_gaze/iter_gaze
                            )
                            )

                # Save models at numbered epochs.
                if epoch % 1 == 0 and epoch < num_epochs:
                    print('Taking snapshot...',
                        torch.save(model.state_dict(),
                                    output+'/fold' + str(fold) +'/'+
                                    '_epoch_' + str(epoch+1) + '.pkl')
                        )


   
    elif data_set=="eyediap":
        # Mutil

        model = Multinet_eye(yaw_num_bins=84, pitch_num_bins=84)        
        model.cuda(gpu)
        dataset=datasets.EyeDiap_multi(args.gazeEyeDiap_dir, args.gazeEyeDiaplabel_dir, transformations, 42)

        print('Loading data.')
        print('len trainData:',len(dataset))
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=False)
        torch.backends.cudnn.benchmark = True

        val_label = "datasets/EYEDIAP/ClusterLabel/3/Cluster2.label"
        val_dataset=datasets.EyeDiap_multi(args.gazeEyeDiap_dir, val_label, transformations, 42)
        print('Loading val data.')
        print('len valData:',len(val_dataset))
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=False)
        torch.backends.cudnn.benchmark = True

        summary_name = '{}'.format('Muti-eyediap-3')
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)

        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(84)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

        # Optimizer gaze
        optimizer_gaze = optim.Adam(model.parameters(), lr=lr)
    
        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}--------------\n"
        print(configuration)
            
        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
        
            for i, (images_gaze, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)

                # Binned labels
                label_pitch = Variable(labels[:, 0]).cuda(gpu)
                label_yaw = Variable(labels[:, 1]).cuda(gpu)

                label_yaw_0 = Variable(labels_0[:,0]).cuda(gpu)
                label_pitch_0 = Variable(labels_0[:,1]).cuda(gpu)

                label_yaw_1 = Variable(labels_1[:,0]).cuda(gpu)
                label_pitch_1 = Variable(labels_1[:,1]).cuda(gpu)
                
                label_yaw_2 = Variable(labels_2[:,0]).cuda(gpu)
                label_pitch_2 = Variable(labels_3[:,1]).cuda(gpu)
                
                label_yaw_3 = Variable(labels_3[:,0]).cuda(gpu)
                label_pitch_3 = Variable(labels_3[:,1]).cuda(gpu)


                # Continuous labels
                label_pitch_cont = Variable(cont_labels[:, 0]).cuda(gpu)
                label_yaw_cont = Variable(cont_labels[:, 1]).cuda(gpu)

                pre_pitch, pre_pitch_0, pre_pitch_1, pre_pitch_2, pre_pitch_3, pre_yaw, pre_yaw_0, pre_yaw_1, pre_yaw_2, pre_yaw_3 = model(images_gaze)

                # Cross entropy loss
                loss_pitch,loss_pitch_0,loss_pitch_1,loss_pitch_2,loss_pitch_3 = criterion(pre_pitch, label_pitch),criterion(pre_pitch_0, label_pitch_0),criterion(pre_pitch_1, label_pitch_1),criterion(pre_pitch_2, label_pitch_2),criterion(pre_pitch_3, label_pitch_3)
                loss_yaw,loss_yaw_0,loss_yaw_1,loss_yaw_2,loss_yaw_3 = criterion(pre_yaw, label_yaw),criterion(pre_yaw_0, label_yaw_0),criterion(pre_yaw_1, label_yaw_1),criterion(pre_yaw_2, label_yaw_2),criterion(pre_yaw_3, label_yaw_3)

                # MSE loss
                pitch_predicted = softmax(pre_pitch)
                yaw_predicted = softmax(pre_yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 1 - 42
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 1 - 42

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont)                

                # one batch Total loss
                loss_pitch_gaze = alpha * loss_reg_pitch + 7*loss_pitch + 4*loss_pitch_0 + 2*loss_pitch_1 + 0.8*loss_pitch_2 + 0.8*loss_pitch_3
                loss_yaw_gaze = alpha * loss_reg_yaw + 7*loss_yaw + 4*loss_yaw_0 + 2*loss_yaw_1 + 0.8*loss_yaw_2 + 0.8*loss_yaw_3

                # all batch loss
                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = \
                    [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()

                iter_gaze += 1

                if (i+1) % 500 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )


            val_loss=[]
            epoch_list=[]
            avg_MAE=[]
            total = 0
            avg_error = .0

            with torch.no_grad():
                for j, (images, labels, labels_0, labels_1, labels_2, labels_3, cont_labels, name)  in enumerate(val_loader):
                    images = Variable(images).cuda(gpu)
                    total += cont_labels.size(0)

                    #角度转化为弧度
                    label_pitch = cont_labels[:,0].float()*np.pi/180
                    label_yaw = cont_labels[:,1].float()*np.pi/180
                    
                    pre_pitch, pre_pitch_0, pre_pitch_1, pre_pitch_2, pre_pitch_3, pre_yaw, pre_yaw_0, pre_yaw_1, pre_yaw_2, pre_yaw_3 = model(images)

                    # Binned predictions
                    _, pitch_bpred = torch.max(pre_pitch.data, 1)
                    _, yaw_bpred = torch.max(pre_yaw.data, 1)
                    
        
                    # Continuous predictions
                    pitch_predicted = softmax(pre_pitch)
                    yaw_predicted = softmax(pre_yaw)
                    
                    # mapping from binned (0 to 28) to angels (-42 to 42)                
                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 1 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 1 - 42
                    
                    #角度转化为弧度
                    pitch_predicted = pitch_predicted*np.pi/180
                    yaw_predicted = yaw_predicted*np.pi/180

                    for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                        error = angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                        avg_error += error
                        val_loss.append(error)
                    
                arr_var = np.var(val_loss)
                arr_std = np.std(val_loss)
                np_avg = np.mean(val_loss)
                print("Val angular:",avg_error/ total)
                print("numpy angular:",np_avg)
                print("len val_loss:",len(val_loss))
                print("Val arr_var:",arr_var)
                print("Val arr_std:",arr_std)
                
                torch.save(model.state_dict(),
                          output +'/'+'epoch_' + str(epoch+1) + '.pkl')




    elif data_set=="rtgene":         
        model = R_WGENet(yaw_num_bins=38, pitch_num_bins=38)  
        model.cuda(gpu)
        
        dataset=datasets.RTGene(args.gazeRTgene_dir, args.gazeRTgenelabel_dir, transformations, 38)
        print('Loading data.')
        print('len trainData:',len(dataset))
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=False)
        torch.backends.cudnn.benchmark = True

        val_label = "datasets/RT-GENE/Label/train/valid.label"
        val_dataset=datasets.RTGene(args.gazeRTgene_dir, val_label, transformations, 38)
        print('Loading val data.')
        print('len valData:',len(val_dataset))
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=False)
        torch.backends.cudnn.benchmark = True


        summary_name = '{}'.format('L2CS-rtgene')
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)

        
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(38)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
        

        # Optimizer gaze
        optimizer_gaze = optim.Adam(model.parameters(), lr=lr)
       

        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)
        
        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

            
            for i, (images_gaze, labels_gaze, cont_labels_gaze, name) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)
                
                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                pitch, yaw = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 2 - 38
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 2 - 38

                loss_reg_pitch = reg_criterion(
                   pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(
                   yaw_predicted, label_yaw_cont_gaze)
                
                # loss_reg_pitch = wrapLoss(
                #     pitch_predicted, label_pitch_cont_gaze,range = 80).cuda(gpu)
                # loss_reg_yaw = wrapLoss(
                #     yaw_predicted, label_yaw_cont_gaze,range = 80).cuda(gpu)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()
                # scheduler.step()
                 
                iter_gaze += 1

                if (i+1) % 500 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' '\n'% (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )

            val_loss=[]
            epoch_list=[]
            avg_MAE=[]
            total = 0
            avg_error = .0

            with torch.no_grad():
                for j, (images, labels, cont_labels, name) in enumerate(val_loader):
                    images = Variable(images).cuda(gpu)
                    total += cont_labels.size(0)

                    #角度转化为弧度
                    label_pitch = cont_labels[:,1].float()*np.pi/180
                    label_yaw = cont_labels[:,0].float()*np.pi/180
                    

                    gaze_yaw, gaze_pitch = model(images)
                    
                    # Binned predictions
                    _, pitch_bpred = torch.max(gaze_pitch.data, 1)
                    _, yaw_bpred = torch.max(gaze_yaw.data, 1)
                    
        
                    # Continuous predictions
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    
                    # mapping from binned (0 to 28) to angels (-42 to 42)                
                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 2 - 38
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 2 - 38
                    
                    #角度转化为弧度
                    pitch_predicted = pitch_predicted*np.pi/180
                    yaw_predicted = yaw_predicted*np.pi/180

                    for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                        error = angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                        avg_error += error
                        val_loss.append(error)    
                    
                arr_var = np.var(val_loss)
                arr_std = np.std(val_loss)
                np_avg = np.mean(val_loss)
                print("Val angular:",avg_error/ total)
                print("numpy angular:",np_avg)
                print("len val_loss:",len(val_loss))
                print("Val arr_var:",arr_var)
                print("Val arr_std:",arr_std)
                
                if np_avg<best_accuracy:
                    best_accuracy = np_avg
                    print('Taking snapshot...',
                        torch.save(model.state_dict(),
                                    output + 'best_epoch_' + str(epoch+1) + '.pkl')
                        )

