import torch
import json
import numpy as np
from torch.nn import functional as F
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms, datasets
from pythae.data.datasets import CelebADataset, TeapotsDataset_with_labels, Dataset
from sklearn.utils import shuffle
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pythae.models import AutoModel


class downstream_task_classification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(downstream_task_classification, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

ap = argparse.ArgumentParser()

ap.add_argument(
    "--dataset",
    help='dataset',
    type=str,
    default=None,
)

ap.add_argument(
    "--model_name",
    help='name of the model',
    type=str,
    default=None,
)

ap.add_argument(
    "--exp_name",
    help='name of the model',
    type=str,
    default=None,
)


ap.add_argument(
    "--latent_dim",
    help='latent dim',
    type=int,
    default=10,
)

ap.add_argument(
    "--data_path",
    help='path to data',
    type=str,
    default='/home/cristianmeo/Datasets/',
)

args = ap.parse_args()


def main(args):
   

    if args.dataset == "celeba":   #[done]

        # Spatial size of training images, images are resized to this size.
        image_size = 64
        img_folder=args.data_path+'celeba/img_align_celeba'
        # Transformations to be applied to each individual image sample

        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])
        # Load the dataset from file and apply transformations
        data = CelebADataset(f'{img_folder}/img_align_celeba', transform)
        train_data = np.zeros((162770, 3, 64, 64), float)
        eval_data = np.zeros((182637 - 162770, 3, 64, 64), float)
        train_labels = np.zeros((162770, 40), float)
        eval_labels = np.zeros((182637 - 162770, 40), float)
        
        with open(f'{img_folder}/list_attr_celeba.txt') as f:
            lines = f.readlines()

            for i in range(162770):
                train_data[i] = data[i]
                label = list(filter(None, lines[i][10:-1].split(' ')))
                for j in range(40):
                    train_labels[i, j] = label[j]

            for j in range(182637 - 162770):
                eval_data[j] = data[162770 + j]
                label = list(filter(None, lines[162770 + j][10:-1].split(' ')))
                for k in range(40):
                    eval_labels[j, k] = label[k]

        train_labels = (train_labels + 1) / 2
        eval_labels = (eval_labels + 1) / 2
        num_classes = 40
        task = 'classification'
        print('data loading done!')

        
      


        
    if args.dataset == "cifar10":  #[done]

        image_size=64
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(image_size)])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        train_data = train_dataset.data.transpose((0, 3, 1, 2))/255

        eval_dataset =  datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        eval_data = eval_dataset.data.transpose((0, 3, 1, 2))/255

        train_labels = np.zeros((50000, 10), float)
        eval_labels = np.zeros((10000, 10), float)
        for i in range(50000):
            train_labels[i, train_dataset.targets[i]] += 1
        for i in range(10000):
            eval_labels[i, eval_dataset.targets[i]] += 1
        num_classes = 10
        task = 'classification'


    if args.dataset == "3Dshapes": 

        dataset = h5py.File(args.data_path+'3dshapes.h5', 'r')

        data =  np.array(dataset['images']).transpose((0, 3, 1, 2))/ 255.0
        labels = np.array(dataset['labels'])
        labels[:, 5] = (labels[:, 5] + 30)/60
        labels[:, 3] = (labels[:, 3] - 0.75)/0.5
        labels[:, 4] = labels[:, 4]/3
        data_n_split = int(data.shape[0]*0.8)
        train_data = data[:data_n_split]
        eval_data = data[data_n_split:]
        train_labels = labels[:data_n_split]
        eval_labels = labels[data_n_split:]

        eval_data_n_split = int(data.shape[0] - data.shape[0]*0.8)
        num_classes= labels.shape[1]
        data_dim = data.shape[0]
        task = 'segmentation'

    if args.dataset == "teapots":

        img_folder=args.data_path+'teapots/'

        # Load the dataset from file and apply transformations
        data = TeapotsDataset_with_labels(f'{img_folder}')
        _, label_example = data[0]
        num_classes = label_example.shape[0]
        train_data = np.zeros((160000, 3, 64, 64), float)
        eval_data = np.zeros((40000, 3, 64, 64), float)
        train_labels = np.zeros((160000, num_classes), float)
        eval_labels = np.zeros((40000, num_classes), float)
        data_n_split = 160000
        
        
        for i in range(160000):
            train_data[i], train_labels[i] = data[i]
        for j in range(40000):
            eval_data[j], eval_labels[j] = data[160000 + j]
        data_dim = train_data.shape[0]
        
        print('data loading done!')

        task = 'segmentation'
        
    train_dataset = Dataset(train_data, train_labels)
    eval_dataset = Dataset(eval_data, eval_labels)

   
    device = "cuda"
   
    model_path = '/home/cristianmeo/benchmark_VAE/reproducibility/'+str(args.dataset)+'/'+str(args.exp_name)+'/final_model'
    loaded_model = AutoModel.load_from_folder(
        model_path
    ).to(device)
    loaded_model.eval()

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # Instantiate the downstream task classifier
    if task == 'classification':
        downstream_model = downstream_task_classification(input_size=args.latent_dim, num_classes=num_classes).to(device)
            # Define the optimizer and loss function
        optimizer = torch.optim.Adam(downstream_model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        # Prepare the data loaders
        

        # Define the number of training epochs
        num_epochs = 10
        
        # Training loop
        for epoch in range(num_epochs):
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()

                # Compute the input for the classifier
                x = {'data': inputs.to(device)}
                input_latent = loaded_model(x)
                output = downstream_model(input_latent.z)
                loss = loss_fn(torch.sigmoid(output), labels.to(device))
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

        # Evaluation on validation dataset
        downstream_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_dataloader:
                x = {'data': inputs.to(device)}
                inputs_latent = loaded_model(x)
                outputs = downstream_model(inputs_latent.z)
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            val_acc = correct / total
            print(f'Validation Acc: {val_acc:.4f}')

                # Save the final accuracy in a JSON file
        results = {'model_name': args.exp_name, 'downstream_classification_accuracy': val_acc}
        if os.path.exists(f'experiments/downstream_classification_tasks/{args.dataset}/results.json'):
            with open(f'experiments/downstream_classification_tasks/{args.dataset}/results.json', "r") as jsonFile:
                data = json.load(jsonFile)
            data.append(results)
            with open(f'experiments/downstream_classification_tasks/{args.dataset}/results.json', "w+") as jsonFile:
                json.dump(data, jsonFile)
        else:
            with open(f'experiments/downstream_classification_tasks/{args.dataset}/results.json', "w+") as jsonFile:
                json.dump([results], jsonFile)
    else: 
        latent_vectors = torch.tensor(np.zeros((32000, args.latent_dim), float), device=device)
      
        latent_vectors_val = torch.tensor(np.zeros((3200, args.latent_dim), float), device=device)
        true_labels_val = np.zeros((3200, num_classes), float)
        i = 0
        for inputs, labels in train_dataloader:
            # Compute the input for the classifier
            x = {'data': inputs.to(device)}
            latent_vectors[32*i: 32*i + 32] = loaded_model(x).z
           
            i = i + 1
            if i==1000:
                break
        i = 0
        for inputs, labels in val_dataloader:
            # Compute the input for the classifier
            x = {'data': inputs.to(device)}
            latent_vectors_val[32*i: 32*i + 32] = loaded_model(x).z
            true_labels_val[32*i: 32*i + 32] = labels
            i = i + 1
            if i==100:
                break


        downstream_model  = KMeans(n_clusters=int(train_labels.shape[1])).fit(latent_vectors.detach().cpu().numpy())
        y_pred = downstream_model.fit_predict(latent_vectors_val.detach().cpu().numpy())
        ari = adjusted_rand_score(true_labels_val.reshape(-1), y_pred.reshape(-1))
        print("Adjusted Rand index: {:.3f}".format(ari))        
        

        # Save the final accuracy in a JSON file
        results = {'model_name': args.exp_name, 'Adjusted Rand index': ari}
        if os.path.exists(f'experiments/downstream_segmentation_task/{args.dataset}/results.json'):
            with open(f'experiments/downstream_segmentation_task/{args.dataset}/results.json', "r") as jsonFile:
                data = json.load(jsonFile)
            data.append(results)
            with open(f'experiments/downstream_segmentation_task/{args.dataset}/results.json', "w+") as jsonFile:
                json.dump(data, jsonFile)
        else:
            with open(f'experiments/downstream_segmentation_task/{args.dataset}/results.json', "w+") as jsonFile:
                json.dump([results], jsonFile)

        # Obtain labels for each point in mesh. Use last trained model.
      

        
         
    

    


if __name__ == "__main__":

    main(args)