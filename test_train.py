from argparse import ArgumentParser
from typing import Type
import os

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
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.tensorboard import SummaryWriter

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

    # Use a proportion of the dataset
    try:
        proportion = cfg.dataset.proportion
    except:
        proportion = 1.0

    train_len = int(proportion*len(dataset['train']['vi']))
    val_len = int(proportion*len(dataset['dev']['vi']))
    test_len = int(proportion*len(dataset['test']['vi']))

    train_dataset = HuggingFaceDataset(dataset['train'].select(range(train_len)), tokenizer, cfg.model.context_length)
    val_dataset = HuggingFaceDataset(dataset['dev'].select(range(val_len)), tokenizer, cfg.model.context_length)
    test_dataset = HuggingFaceDataset(dataset['test'].select(range(test_len)), tokenizer, cfg.model.context_length)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    test_loader = DataLoader(test_dataset, batch_size = cfg.training.batch_size)
    print("Finished loading data")

    # Init model
    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))).to(
        device=cfg.training.device
    )
    
    model.reset_parameters()
    if cfg.training.load_checkpoint:
        model.load_state_dict(torch.load(cfg.training.load_checkpoint, map_location = torch.device(cfg.training.device)))
    model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision])
    print("Model initialized")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

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

    writer = SummaryWriter(f'{cfg.training.save_checkpoint}/runs')

    # Training loop
    step = 0
    running_loss = 0.0
    print("Start training\n")
    if not os.path.exists(cfg.training.save_checkpoint):
        os.makedirs(cfg.training.save_checkpoint)

    for epoch in range(1, (cfg.training.num_epochs+1)):
        print(f"Epoch: {epoch}/{cfg.training.num_epochs}")
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), initial=0)
        for inputs, labels in pbar:
            inputs = inputs.to(device=cfg.training.device)
            labels = labels.to(device=cfg.training.device)

            optimizer.zero_grad()
            with torch.autocast(
                device_type=cfg.training.device,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                outputs = model(inputs.to(device=cfg.training.device))
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
                pbar.set_description(f"Training Loss Epoch {epoch}: {running_loss:.4f}")

            step += 1
            if step % cfg.training.log_every_step == 0:
                print(f"Step: {step}, Loss: {running_loss:.4f}")
                torch.save(model.state_dict(), f"{cfg.training.save_checkpoint}/latest.pth")
                torch.save(model.state_dict(), f"{cfg.training.save_checkpoint}/e{epoch}s{step//cfg.training.log_every_step}.pth")
                writer.add_scalar("Loss/train", running_loss, step//cfg.training.log_every_step)
                

        # if step % cfg.training.log_every_step == 0:
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
                    f"Validation Loss Step {step}: {(val_loss/len(val_loader)):.4f}"
                )
                writer.add_scalar("Loss/val", val_loss/len(val_loader), step//cfg.training.log_every_step)
        
        if epoch % cfg.training.test_every_epoch == 0:
            references_list = []
            translations = []
            test_loss = 0.0
            model.eval()
            for inputs, labels in tqdm(test_loader, total=len(test_loader), initial=0):
                test_inputs = inputs.to(device=cfg.training.device)
                test_labels = labels.to(device=cfg.training.device)

                with torch.no_grad():
                    with torch.autocast(
                        device_type=cfg.training.device,
                        dtype=torch_dtype_map[cfg.training.amp_precision],
                        enabled=cfg.training.enable_mixed_precision,
                    ):
                        test_outputs = model(test_inputs)
                        loss = nn.functional.cross_entropy(
                            test_outputs.view(-1, cfg.model.vocab_size),
                            test_labels.view(-1),
                            ignore_index=-1,
                        )
                        # print("Val single loss", loss)
                        test_loss += loss.item()
                        # token_indices = test_outputs.argmax(dim=-1)
                        # candidates = [tokenizer.decode(tokens) for tokens in token_indices]
                        # references = [tokenizer.decode(tokens) for tokens in test_labels]
                        # references = [ref.split('<envi>')[-1].replace("<pad>", "").strip() for ref in references]
                        # candidates = [can.split('<envi>')[-1].replace("<pad>", "").strip() for can in candidates]
                        # for ref in references:
                        #     references_list.append([ref.split(' ')])
                        # for can in candidates:
                        #     translations.append(can.split(' '))
            print(f"Test Loss: {(test_loss/len(test_loader)):.4f}")
            writer.add_scalar("Loss/test", test_loss/len(test_loader), step//cfg.training.log_every_step)
            # bleu_score_corpus = corpus_bleu(references_list, translations)
            # print(f'BLEU score: {bleu_score_corpus}')


        # Save model
        # torch.save(model.state_dict(), f"{cfg.training.save_checkpoint}/epoch{epoch}.pth")
        # print(f"Model saved at {cfg.training.save_checkpoint}", )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default="test_train_cfg.yaml")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg)
