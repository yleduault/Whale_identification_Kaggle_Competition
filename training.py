import os.path
import albumentations as A
import torch
import tqdm
from torch.nn import Linear
from torchvision.models import resnet101
from Config import global_path
from torch.utils.data import DataLoader, random_split
from load_data import WhaleDataset


def extract_params(csv_path,model_idx:int):
    return {}

def create_transform(*args):
    return None


class TrainingSchema:
    def __init__(self,model_params_csv_path):
        # self.random_seed = np.random.random()
        self.models = model_params_csv_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()

    def training(self,model_idx:list):
        torch.cuda.empty_cache()
        if len(model_idx) < 1: raise Exception("No model select for training")
        for id in model_idx:
            # params = extract_params(self.models,id)

            # TODO remove following lines when csv and params extract is done
            params = {
                'transform':None,
                'file_system':
                    {
                    'data_folder':os.path.join(global_path,'humpback-whale-identification/'),
                    'labels_csv':os.path.join(global_path,'humpback-whale-identification/train.csv')
                    },
                'set_split': 0.80,
                'training':
                    {
                        'nb_epochs':10,
                        'batch_size': 32,
                    }

            }

            # transforms = create_transform(params["transform"])

            transforms = A.Compose([
                A.Resize(256,256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2()
            ])


            whale_dataset = WhaleDataset(params['file_system']['data_folder'],params['file_system']['labels_csv'],data_augmentation=transforms)


            train_set, validation_set = random_split(whale_dataset,[params["set_split"],1-params["set_split"]])

            train_dataloader = DataLoader(train_set,batch_size=int(params["training"]['batch_size']))
            validation_dataloader = DataLoader(validation_set)

            # Model loading

            model = resnet101()
            model.fc = Linear(in_features=2048,out_features=len(whale_dataset.id_list))
            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

            # Training loop
            nb_epochs = int(params["training"]['nb_epochs'])
            for epoch in range(nb_epochs):
                model.train()
                train_loss = 0.0
                train_preds, train_targets = [], []
                pbar = tqdm.tqdm(train_dataloader,desc=f"Epoch {epoch+1}/{nb_epochs} [Training]")
                for data, label in pbar:
                    data,label = data.to(self.device),label.to(self.device)
                    scores = model(data)
                    loss = self.criterion(scores, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Compute losses and predictions for global epoch metrics computation
                    train_loss += loss.item()
                    preds = torch.argmax(scores, dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_targets.extend(label.cpu().numpy())

                    pbar.set_postfix(loss=loss.item())


if __name__ == '__main__':
    training_scheme = TrainingSchema('')
    training_scheme.training([0])