from argparse import ArgumentParser
from typing import Type

import torch
from dacite import from_dict
from experiments.data.formal_language.formal_language_dataset import (
    FormLangDatasetGenerator,
)
from experiments.data.utils import DataGen
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from nltk.translate.bleu_score import corpus_bleu

from test_train import load_tokenizer, construct_sample, HuggingFaceDataset

dataset_registry: dict[str, Type[DataGen]] = {
    "form_language": FormLangDatasetGenerator
}

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


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
        propotion = cfg.dataset.proportion
    except:
        propotion = 1.0

    test_len = int(propotion*len(dataset['test']['vi']))

    test_dataset = HuggingFaceDataset(dataset['test'].select(range(test_len)), tokenizer, cfg.model.context_length)

    # Loaders
    test_loader = DataLoader(test_dataset, batch_size = cfg.training.batch_size)
    print("Finished loading data")

    # Init model
    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))).to(
        device=cfg.training.device
    )
    model.reset_parameters()
    model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision])
    model.load_state_dict(torch.load('test_1024.pth', map_location = torch.device(cfg.training.device)))
    print("Model initialized")

        
    model.eval()
    references_list = []
    translations = []
    for inputs, labels in tqdm(test_loader, total=len(test_loader), initial=0):
        test_inputs = inputs.to(device=cfg.training.device)
        test_labels = labels.to(device=cfg.training.device)
        input_lens = []
        for i in range(test_inputs.size(0)):
            end_index = (test_inputs[i] == 50100).nonzero(as_tuple=True)[0]
            if end_index.numel() > 0:
                input_lens.append(end_index.item())
                test_inputs[i, end_index[0] + 1:] = 0

        test_outputs = []
        with torch.no_grad():
            with torch.autocast(
                device_type=cfg.training.device,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                for i, test_input in enumerate(test_inputs):
                    input_len = input_lens[i]
                    temp = test_input
                    test_output = []
                    while True:
                        gen_token = model(temp.unsqueeze(0)).argmax(dim=-1)[0][input_len]
                        test_output.append(gen_token.item())
                        input_len += 1
                        temp[input_len] = gen_token
                        if input_len == 255 or gen_token.item() == 1:
                            test_outputs.append(test_output)
                            break
                
                # test_outputs = model(test_inputs)
                # token_indices = test_outputs.argmax(dim=-1)
                candidates = [tokenizer.decode(tokens) for tokens in test_outputs]
                references = [tokenizer.decode(tokens) for tokens in test_labels]
                references = [ref.split('<envi>')[-1].replace("<pad>", "").strip() for ref in references]
                # candidates = [can.split('<envi>')[-1].replace("<pad>", "").strip() for can in candidates]
                for ref in references:
                    references_list.append([ref])
                for can in candidates:
                    translations.append(can)
    
    import pickle
    with open ('infer_result.pkl', 'wb') as fp:
        pickle.dump(translations, fp)
    bleu_score_corpus = corpus_bleu(references_list, translations)
    print(f'Bleu score: {bleu_score_corpus}')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default="eval.yaml")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg)
