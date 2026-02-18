import os
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from torch.nn import Linear
from torchvision import models
from torch.utils.data import DataLoader, random_split
from Config import global_path
from load_data import WhaleDataset


def extract_params(yaml_path: str, model_idx: int):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    models_config = config.get("models", [])
    if model_idx >= len(models_config):
        raise IndexError(f"Model index {model_idx} not found in YAML")

    return models_config[model_idx]


def create_transform(transform_list: list):
    transform_objects = []

    for transform_cfg in transform_list:
        name = transform_cfg["name"]
        params = transform_cfg.get("params", {})

        # Dynamically fetch Albumentations class
        if hasattr(A, name):
            transform_class = getattr(A, name)
        elif name == "ToTensorV2":
            transform_class = ToTensorV2
        else:
            raise ValueError(f"Unknown transform: {name}")

        transform_objects.append(transform_class(**params))

    return A.Compose(transform_objects)


def create_model(architecture: str, pretrained: bool, num_classes: int):
    model_fn = getattr(models, architecture)
    model = model_fn(pretrained=pretrained)

    if hasattr(model, "fc"):  # For ResNet-like architectures
        in_features = model.fc.in_features
        model.fc = Linear(in_features, num_classes)
    else:
        raise NotImplementedError(f"{architecture} final layer replacement not implemented")

    return model


class TrainingSchema:
    def __init__(self, model_params_yaml_path):
        self.yaml_path = model_params_yaml_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()

    def _train_one_fold(self, model, train_loader, val_loader,
                        optimizer, nb_epochs, fold_id=None):

        for epoch in range(nb_epochs):

            model.train()
            running_loss = 0.0

            # Single active progress bar (batch level only)
            with tqdm(
                    train_loader,
                    desc=f"Fold {fold_id} | Epoch {epoch + 1}/{nb_epochs}",
                    leave=False,
                    dynamic_ncols=True
            ) as pbar:

                for data, label in pbar:
                    data = data.to(self.device)
                    label = label.to(self.device)

                    outputs = model(data)
                    loss = self.criterion(outputs, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}"
                    )

            val_acc = self._evaluate(model, val_loader)

            tqdm.write(
                f"Fold {fold_id} | "
                f"Epoch {epoch + 1}/{nb_epochs} | "
                f"Loss: {running_loss / len(train_loader):.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    def _evaluate(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(self.device), label.to(self.device)

                outputs = model(data)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == label).sum().item()
                total += label.size(0)

        return correct / total

    def training(self, model_indices: list):
        torch.cuda.empty_cache()

        if len(model_indices) < 1:
            raise Exception("No model selected for training")

        for idx in model_indices:

            # --------------------
            # Load YAML Parameters
            # --------------------
            params = extract_params(self.yaml_path, idx)
            cv_params = params.get("cross_validation", {})
            use_cv = cv_params.get("use", False)

            data_folder = os.path.join(global_path, params["file_system"]["data_folder"])
            labels_csv = os.path.join(global_path, params["file_system"]["labels_csv"])

            transforms = create_transform(params["transforms"])


            # --------------------
            # Dataset split
            # --------------------

            if use_cv:
                whale_dataset = WhaleDataset(
                    data_folder,
                    labels_csv,
                    data_augmentation=None  # important: we apply transforms later
                )

                targets = [whale_dataset.paths[i][0] for i in range(len(whale_dataset))]
            else:

                whale_dataset = WhaleDataset(
                    data_folder,
                    labels_csv,
                    data_augmentation=transforms
                )

                split_ratio = params["set_split"]
                train_size = int(split_ratio * len(whale_dataset))
                val_size = len(whale_dataset) - train_size

                train_set, validation_set = random_split(
                    whale_dataset,
                    [train_size, val_size]
                )

                train_loader = DataLoader(
                    train_set,
                    batch_size=params["training"]["batch_size"],
                    shuffle=True
                )

                val_loader = DataLoader(validation_set)


            # --------------------
            # Model creation
            # --------------------
            model = create_model(
                architecture=params["architecture"],
                pretrained=params["pretrained"],
                num_classes=len(whale_dataset.id_list)
            )

            model.to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params["training"]["lr"]
            )

            nb_epochs = params["training"]["nb_epochs"]



            # --------------------
            # Training Loop
            # --------------------


            # Check if cross validation is used
            if use_cv:
                assert targets is not None
                n_folds = cv_params["n_folds"]

                skf = StratifiedKFold(
                    n_splits=cv_params["n_folds"],
                    shuffle=cv_params["shuffle"],
                    random_state=cv_params["random_seed"]
                )
                for fold, (train_idx, val_idx) in enumerate(
                        StratifiedKFold(
                            n_splits=cv_params["n_folds"],
                            shuffle=cv_params["shuffle"],
                            random_state=cv_params["random_seed"]
                        ).split(range(len(whale_dataset)), targets)
                    ):
                    tqdm.write(f"\n========== Fold {fold + 1}/{n_folds} ==========")

                    train_subset = Subset(whale_dataset, train_idx)
                    val_subset = Subset(whale_dataset, val_idx)

                    # Attach transforms independently
                    train_subset.dataset.data_augmentations = create_transform(params["transforms"])
                    val_subset.dataset.data_augmentations = create_transform(params["transforms"])

                    train_loader = DataLoader(
                        train_subset,
                        batch_size=params["training"]["batch_size"],
                        shuffle=True
                    )

                    val_loader = DataLoader(val_subset)

                    model = create_model(
                        params["architecture"],
                        params["pretrained"],
                        len(whale_dataset.id_list)
                    ).to(self.device)

                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=params["training"]["lr"]
                    )

                    self._train_one_fold(
                        model,
                        train_loader,
                        val_loader,
                        optimizer,
                        params["training"]["nb_epochs"],
                        fold_id=fold + 1
                    )

            else:
                for epoch in range(nb_epochs):
                    model.train()
                    train_loss = 0.0

                    with tqdm(
                            train_loader,
                            desc=f"{params['name']} | Epoch {epoch + 1}/{nb_epochs}",
                            leave=False,
                            dynamic_ncols=True
                    ) as pbar:

                        for data, label in pbar:
                            data = data.to(self.device)
                            label = label.to(self.device)

                            scores = model(data)
                            loss = self.criterion(scores, label)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            train_loss += loss.item()
                            pbar.set_postfix(loss=loss.item())

                        tqdm.write(f"Epoch {epoch+1} Loss: {train_loss / len(train_loader):.4f}")

                tqdm.write(f"Finished training model: {params['name']}")


if __name__ == '__main__':
    training_scheme = TrainingSchema("models.yaml")

    # Train multiple models from YAML
    training_scheme.training([0, 1])
