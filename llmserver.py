import torch
from pathlib import Path
from transformers import LongformerConfig, LongformerModel, LongformerTokenizer
from parse import args


class LongformerServer:
    def __init__(self, args):
        self.args = args
        config = LongformerConfig.from_pretrained(self.args.original_pretrain_ckpt, output_hidden_states=True)
        self.tokenizer = LongformerTokenizer.from_pretrained(self.args.original_pretrain_ckpt)
        self.model = LongformerModel.from_pretrained(self.args.original_pretrain_ckpt, config=config)

        self.path_corpus = Path(self.args.data_path)
        self.dir_preprocess = self.path_corpus / 'preprocess'
        self.dir_preprocess.mkdir(exist_ok=True)

    def encode_dict(self, a_dict):
        self.model.eval()
        return_dict = {}
        with torch.no_grad():
            for iid, text in a_dict.items():
                encoding = self.tokenizer(text, return_tensors="pt").to(args.device_id)
                global_attention_mask = [1].extend([0] * encoding["input_ids"].shape[-1])
                encoding["global_attention_mask"] = global_attention_mask
                output = self.model(**encoding)
                setence_embedding = output.last_hidden_state[:, 0].detach()
                return_dict[iid] = setence_embedding.cpu()
        return return_dict

    def encode_items(self, item_texts):
        path_item_embeddings = self.dir_preprocess / f'{self.__class__.__name__}_items_{self.path_corpus.name}'
        if path_item_embeddings.exists():
            print(f'======[Preprocessor] Use cache: {path_item_embeddings}')
        else:
            item_to_semantic = self.encode_dict(item_texts)
            tmp = []
            for i in range(len(item_to_semantic)):
                tmp.append(item_to_semantic[i])
            item_to_semantic = torch.cat(tmp, dim=0)
            torch.save(item_to_semantic, path_item_embeddings)
        item_embeddings = torch.load(path_item_embeddings)
        return item_embeddings
