from argparse import ArgumentParser
from typing import Type

import torch
import torch.optim as optim
from dacite import from_dict
from experiments.data.formal_language.formal_language_dataset import (
    FormLangDatasetGenerator,
)
from experiments.data.utils import DataGen
from experiments.lr_scheduler import LinearWarmupCosineAnnealing
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

dataset_registry: dict[str, Type[DataGen]] = {
    "form_language": FormLangDatasetGenerator
}

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_tokenizer(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    # Add new token for translation
    new_token = "<envi>"
    tokenizer.add_tokens([new_token])

    return tokenizer


def construct_sample(tokenizer, en_text, vi_text, context_length):
    out = tokenizer.encode(f"{en_text}<envi>{vi_text}",
                           padding='max_length',  # Enable padding
                           truncation=True,  # Optional: truncate to max_length if necessary
                           max_length=context_length,  # Optional: specify max length
                           # return_tensors='pt'  # Return PyTorch tensors; use 'tf' for TensorFlow
                           )
    pads = [0] * (context_length - len(out)+1)
    input_ids = torch.tensor(out[:-1] + pads)
    output_ids = torch.tensor(out[1:] + pads)
    return input_ids, output_ids


class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, context_length):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert the Hugging Face dataset item to PyTorch tensor
        item = self.dataset[idx]
        en_text, vi_text = item['en'], item['vi']
        input_ids, labels = construct_sample(self.tokenizer, en_text, vi_text, self.context_length)
        return input_ids, labels


def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed)

    # Load tokenizer
    tokenizer = load_tokenizer(cfg)
    print(f"Tokenizer loaded: {cfg.tokenizer}")

    # Load data
    dataset = load_dataset(cfg.dataset.name)
    # train_test_split = dataset['train'].train_test_split(test_size=0.1)
    smaller_dataset = dataset['train'].train_test_split(test_size=0.01)['test']
    train_test_split = smaller_dataset.train_test_split(test_size=0.1)

    train_dataset = HuggingFaceDataset(train_test_split['train'], tokenizer, cfg.model.context_length)
    test_dataset = HuggingFaceDataset(train_test_split['test'], tokenizer, cfg.model.context_length)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size)
    val_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size)
    print("Finished loading data")
    # import ipdb; ipdb.set_trace()

    # Init model
    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))).to(
        device=cfg.training.device
    )
    model.reset_parameters()
    model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision])
    print("Model initialized")

    # Load optimizer
    optim_groups = model._create_weight_decay_optim_groups()
    optimizer = optim.AdamW(
        (
            {"weight_decay": cfg.training.weight_decay, "params": optim_groups[0]},
            {"weight_decay": 0.0, "params": optim_groups[1]},
        ),
        lr=cfg.training.lr,
    )

    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        cfg.training.lr_warmup_steps,
        cfg.training.lr_decay_until_steps,
        cfg.training.lr,
        cfg.training.lr_decay_factor * cfg.training.lr,
    )

    # Training loop
    step = 0
    running_loss = 0.0
    print("Start training\n")

    for epoch in range(1, (cfg.training.num_epochs+1)):
        print(f"Epoch: {epoch}/{cfg.training.num_epochs}")
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), initial=0)
        for inputs, labels in pbar:
            # import ipdb; ipdb.set_trace()
            inputs = inputs.to(device=cfg.training.device)
            labels = labels.to(device=cfg.training.device)

            optimizer.zero_grad()
            with torch.autocast(
                device_type=cfg.training.device,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):

                outputs = model(inputs.to(device=cfg.training.device))
                # import ipdb; ipdb.set_trace()
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, cfg.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-1,
                )
                # print("singe loss", loss)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                running_loss = running_loss*step / (step+1) + loss.item() / (step + 1)
                pbar.set_description(f"Training Loss: {running_loss:.4f}")

            step += 1

        if epoch % cfg.training.val_every_epoch == 0:
            val_loss = 0.0
            model.eval()
            for inputs, labels in tqdm(val_loader, total=len(val_loader), initial=0):
                val_inputs = inputs.to(device=cfg.training.device)
                val_labels = labels.to(device=cfg.training.device)

                with torch.no_grad():
                    with torch.autocast(
                        device_type=cfg.training.device,
                        dtype=torch_dtype_map[cfg.training.amp_precision],
                        enabled=cfg.training.enable_mixed_precision,
                    ):
                        val_outputs = model(val_inputs)
                        loss = nn.functional.cross_entropy(
                            val_outputs.view(-1, cfg.model.vocab_size),
                            val_labels.view(-1),
                            ignore_index=-1,
                        )
                        # print("Val single loss", loss)
                        val_loss += loss.item()
            print(
                f"Validation Loss: {(val_loss/len(val_loader)):.4f}"
            )
        print("")

    # Save model
    torch.save(model.state_dict(), cfg.training.save_checkpoint)
    print(f"Model saved at {cfg.training.save_checkpoint}", )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default="test_train_cfg.yaml")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg)
