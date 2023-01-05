import torch 
import torch.nn as nn 
from convit import Convit
from torchvision import datasets, transforms 
import torch.backends.cudnn as cudnn 
from torch.utils import data
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy
import matplotlib.pyplot as plt  

from utils import clip_gradient
import warmup_scheduler
import os 



def train_func(train_loader, model, optimizer, loss_func, max_epochs = 100,  
                validation_loader = None, batch_size = 128, scheduler = None, device = None, test_loader = None, 
                train_loader_plain = None, clip_grad = 2.0, path = None, mixup_fn = None):

    """Training function for ConViT.


    Parameters
    ----------
    train_loader: Instance of `torch.utils.data.DataLoader`

    model: Instance of `torch.nn.Module'
            The Vision Transformer as the model model.


    optimizer: Instance of `torch.optim`
            Optimizer for training.

    loss_func: Instance of `torch.nn.Module'
            Loss function for the training.


    Returns
    -------
    history: dict 
            Returns training and validation loss. 

    """

    n_batches_train = len(train_loader)
    n_batches_val = len(validation_loader)
    n_samples_train = batch_size * n_batches_train
    n_samples_val = batch_size * n_batches_val


    losses = []
    accuracy = []
    validation_loss = []
    validation_accuracy = []


    for epoch in range(max_epochs):
        running_loss, correct = 0, 0
        for images, labels in train_loader:
            if device:
                images = images.to(device)
                labels = labels.to(device)
           
            if mixup_fn: 
                images, labels = mixup_fn(images, labels)

            #================= Training ======================
            model.train()
            with torch.cuda.am.autocast():
                outputs = model(images)
                loss = loss_func(outputs, labels)
            
            predictions = outputs.argmax(1)
            correct += int(sum(predictions == labels))
            running_loss += loss.item()


            #================= BACKWARD AND OPTIMZIE  ====================================   
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(model, clip_grad)
            optimizer.step()


        loss_epoch = running_loss / n_batches_train
        accuracy_epoch = correct / n_samples_train
        
        scheduler.step()
        losses.append(loss_epoch)
        accuracy.append(accuracy_epoch)


        print('Epoch [{}/{}], Training Accuracy [{:.4f}], Training Loss: {:.4f}'
            .format(epoch + 1, max_epochs, accuracy_epoch, loss_epoch), end = '  ')
        print('Correct/ Total: [{}/{}]'.format(correct, n_samples_train), end = '   ')


        #====================== Validation ============================
        if validation_loader:
            model.eval()   

            val_loss, val_corr = 0, 0
            for val_images, val_labels in validation_loader:
                if device:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                outputs = model(val_images)
                loss = loss_func(outputs, val_labels)
                _, predictions = outputs.max(1)
                val_corr += int(sum(predictions == val_labels))
                val_loss += loss.item()


            loss_val = val_loss / n_batches_val
            accuracy_val = val_corr / n_samples_val

            validation_loss.append(loss_val)
            validation_accuracy.append(accuracy_val)

            print('Validation accuracy [{:.4f}], Validation Loss: {:.4f}'
            .format(accuracy_val, loss_val))


        #====================== Saving the Model ============================  
        model_save_name = 'model.pt'
        path_model = F"{path}/{model_save_name}"
        torch.save(model.state_dict(), path_model)
    
    #====================== Testing ============================      
    if test_loader:
        correct = 0
        total = 0

        for images, labels in test_loader:
            if device:
                images = images.to(device)
                labels = labels.to(device)

            n_data = images[0]
            total += n_data
            outputs = model(images)
            predictions = outputs.argmax(1)
            correct += int(sum(predictions == labels))

        accuracy = correct / total 
        print('Test Accuracy: {}'.format(accuracy))


    return  model, {'loss': losses, 'accuracy': accuracy, 
            'val_loss': validation_loss, 'val_accuracy': validation_accuracy}



def main(parameters):

    #=============================Preparing Data==================================
    cudnn.benchmark = True
    path = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Starting Convit ...')
    print(device)
    plain_augmentation = transforms.Compose([
        #transforms.Resize(32),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    dataset_train = datasets.CIFAR10(path, download = True, train = True, transform = plain_augmentation)
    dataset_test = datasets.CIFAR10(path, download = True, train = False, transform = plain_augmentation)
    dataset_train_evaluation = datasets.CIFAR10(path, download = True, train = True, transform = plain_augmentation)
    dataset_validation, dataset_test = torch.utils.data.random_split(dataset_test, [8000, 2000])


    train_loader = data.DataLoader(dataset_train, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)
    val_loader = data.DataLoader(dataset_validation, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)
    train_loader_plain = data.DataLoader(dataset_train_evaluation, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)


    #=============================Preparing The Model==================================
    model = Convit(img_size = parameters['img_size'], patch_size = parameters['patch_size'], 
        n_classes = parameters['n_classes'], embed_dim = parameters['embed_dim'], 
        n_heads = parameters['n_heads'], mlp_ratio = parameters['mlp_ratio'], qkv_bias = parameters['qkv_bias'], 
        drop = parameters['drop'], attn_drop = parameters['attn_drop'], local_layers = parameters['local_layers'], 
        locality_strength = parameters['locality_strength'], depth = parameters['depth'], use_pos_embed = parameters['use_pos_embed'])
    
    n_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('The number of trainable parameters is : {}'.format(n_parameters))

    model = model.to(device)
    
    if parameters['mixup']:
        mixup = Mixup(mixup_alpha = parameters['mixup'], cutmix_alpha = parameters['cutmix'], cutmix_minmax = parameters['cutmix_minmax'],
                prob = parameters['prob'], switch_prob=parameters['switch_prob'], mode = parameters['mode'],
                label_smoothing = parameters['label_smoothing'], num_classes = parameters['n_classes']) 

    criterion = nn.CrossEntropyLoss().to(device)

    if parameters['label_smoothing']:
        criterion = LabelSmoothingCrossEntropy(smoothing = parameters['label_smoothing']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = parameters['lr'], weight_decay = parameters['weight_decay'])
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-6)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler = base_scheduler)
    
    model, history = train_func(train_loader = train_loader, model = model, 
        optimizer = optimizer, loss_func = criterion, validation_loader = val_loader, 
        device = device, scheduler = scheduler, batch_size = parameters['batch_size'], 
        max_epochs = parameters['max_epochs'], train_loader_plain = train_loader_plain, 
        clip_grad = parameters['clip_grad'], path = path, mixup_fn = mixup)


    return model, history




if __name__ == '__main__':
    #Replace Parameters with argparse immediately
    parameters = {'batch_size': 512, 'lr': 0.0005, 'weight_decay': 0.05, 'img_size': 32, 
                'n_heads' : 8, 'patch_size' : 8, 'n_classes' : 10, 
                'embed_dim' : 384, 'max_epochs' : 100, 'clip_grad': 2.0, 
                'mlp_ratio': 4, 'qkv_bias': False, 'drop': 0., 'attn_drop': 0., 'local_layers':10, 
                'locality_strength': 1., 'depth': 12, 'use_pos_embed': True,'mixup': 0.8, 
                'cutmix':1 , 'cutmix_minmax':None, 'prob': 1, 'switch_prob': 0.5, 'mode': 'batch', 
                'label_smoothing':0.1}

    model, history = main(parameters)
    
    

    #=============================Validation & Visualizing Embeddings ==================================
    




