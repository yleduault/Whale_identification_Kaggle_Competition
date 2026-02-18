import os
import yaml
from torch.utils.data import WeightedRandomSampler, DataLoader, Subset
from collections import Counter
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.nn import Linear
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from load_data import WhaleDataset
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
        # self.device = torch.device("cpu") # For testing purpose
        self.criterion = None  # Will set per-dataset/fold

    # -------------------------
    # Helper: Class weights
    # -------------------------
    def _compute_class_weights(self, targets,num_classes,device):
        counts = Counter(targets)
        weights = []
        total_samples = len(targets)
        for i in range(num_classes):
            if i in counts:
                weights.append(total_samples / counts[i])
            else:
                weights.append(0.0)  # class not present in this fold/subset
        return torch.tensor(weights, dtype=torch.float).to(device)

    # -------------------------
    # Helper: Weighted sampler
    # -------------------------
    def _create_sampler(self, targets):
        counts = Counter(targets)
        sample_weights = [1.0 / counts[t] for t in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return sampler

    # -------------------------
    # Train one fold
    # -------------------------
    def _train_one_fold(self, model, train_loader, val_loader, optimizer, nb_epochs, fold_id=None):
        for epoch in range(nb_epochs):
            model.train()
            running_loss = 0.0

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
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_acc = self._evaluate(model, val_loader)
            tqdm.write(
                f"\nFold {fold_id} | Epoch {epoch + 1}/{nb_epochs} | "
                f"Loss: {running_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}"
            )

    # -------------------------
    # Evaluate accuracy
    # -------------------------
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

    # -------------------------
    # Training entry point
    # -------------------------
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
            # Dataset & targets
            # --------------------
            whale_dataset = WhaleDataset(data_folder, labels_csv, data_augmentation=None)
            targets = [whale_dataset.paths[i][0] for i in range(len(whale_dataset))]

            num_classes = max(targets) + 1
            # --------------------
            # Cross-validation
            # --------------------
            if use_cv:
                n_folds = cv_params["n_folds"]
                for fold, (train_idx, val_idx) in enumerate(
                        KFold(
                            n_splits=n_folds,
                            shuffle=cv_params["shuffle"],
                            random_state=cv_params["random_seed"]
                        ).split(range(len(whale_dataset)))
                ):
                    tqdm.write(f"\n========== Fold {fold + 1}/{n_folds} ==========")

                    train_subset = Subset(whale_dataset, train_idx)
                    val_subset = Subset(whale_dataset, val_idx)

                    # Apply transforms
                    train_subset.dataset.data_augmentations = create_transform(params["transforms"])
                    val_subset.dataset.data_augmentations = create_transform(params["transforms"])

                    # Weighted loss & sampler
                    fold_targets = [targets[i] for i in train_idx]
                    class_weights = self._compute_class_weights(fold_targets, num_classes, self.device)
                    self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                    sampler = self._create_sampler(fold_targets)

                    train_loader = DataLoader(
                        train_subset,
                        batch_size=params["training"]["batch_size"],
                        sampler=sampler
                    )
                    val_loader = DataLoader(val_subset)

                    # Model & optimizer
                    model = create_model(
                        params["architecture"],
                        params["pretrained"],
                        num_classes
                    ).to(self.device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=params["training"]["lr"])

                    self._train_one_fold(model, train_loader, val_loader, optimizer,
                                         params["training"]["nb_epochs"], fold_id=fold + 1)

            # --------------------
            # Non-CV training
            # --------------------
            else:
                split_ratio = params["set_split"]
                train_size = int(split_ratio * len(whale_dataset))
                val_size = len(whale_dataset) - train_size
                train_set, validation_set = torch.utils.data.random_split(
                    whale_dataset, [train_size, val_size]
                )

                # Weighted loss & sampler
                train_targets = [targets[i] for i in train_set.indices]
                class_weights = self._compute_class_weights(train_targets, num_classes, self.device)
                self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                sampler = self._create_sampler(train_targets)

                train_loader = DataLoader(
                    train_set,
                    batch_size=params["training"]["batch_size"],
                    sampler=sampler
                )
                val_loader = DataLoader(validation_set)

                # Model & optimizer
                model = create_model(
                    params["architecture"],
                    params["pretrained"],
                    num_classes
                ).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=params["training"]["lr"])

                # Train
                self._train_one_fold(model, train_loader, val_loader, optimizer,
                                     params["training"]["nb_epochs"], fold_id=None)

                tqdm.write(f"\nFinished training model: {params['name']}\n")


if __name__ == '__main__':
    training_scheme = TrainingSchema("models.yaml")

    # Train multiple models from YAML
    training_scheme.training([0, 1])
