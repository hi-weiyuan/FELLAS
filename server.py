from parse import args
from models import *
from llmserver import LongformerServer


class FedRecServer(nn.Module):
    def __init__(self, config, item2id, item_meta_dict):
        super().__init__()

        self.llm_server = LongformerServer(args)
        self.llm_server.model.to(args.device_id)
        self.item_id_to_text = {id: item_meta_dict[item]["title"] for item, id in item2id.items()}
        self.word_item_embed = self.llm_server.encode_items(self.item_id_to_text)
        self.model = Model(config, self.word_item_embed)

        self.config = config

    def aggregate_gradients(self, clients_grads):
        clients_num = len(clients_grads)
        aggregated_gradients = {}
        for uid, grads_dict in clients_grads.items():
            for name, grad in grads_dict.items():
                if name in aggregated_gradients:
                    aggregated_gradients[name] = aggregated_gradients[name] + grad / clients_num
                else:
                    aggregated_gradients[name] = grad / clients_num

        old_model_state = self.model.state_dict()
        for name, gradient in aggregated_gradients.items():
            old_model_state[name]  = old_model_state[name] + gradient
        return old_model_state

    def train_(self, clients, batch_clients_idx, llm_u_to_seqvector, neg_llm_u_to_seqvector):
        self.model.train()

        client_grads, batch_loss = clients.train_(batch_clients_idx, self.model.state_dict(), llm_u_to_seqvector, neg_llm_u_to_seqvector)

        new_model_state = self.aggregate_gradients(client_grads)
        self.model.load_state_dict(new_model_state)

        return batch_loss

    def call_llm_help(self, client):
        prompt = "The user's purchase history list is as follows: "

        train, val, test = client.query()
        u_to_seq_text = {}
        for u, content in train.items():
            u_to_seq_text[u] = prompt + ". ".join([self.item_id_to_text[iid] for iid in content[::-1][:self.config.max_len]]) + "."
        llm_u_to_seqvector = self.llm_server.encode_dict(u_to_seq_text)
        neg_train = client.query_for_neg()
        u_to_neg_seq_text = {}
        for u, content in neg_train.items():
            u_to_neg_seq_text[u] = prompt + ". ".join([self.item_id_to_text[iid] for iid in content[::-1][:self.config.max_len]])
        neg_llm_u_to_seqvector = self.llm_server.encode_dict(u_to_neg_seq_text)

        return llm_u_to_seqvector, neg_llm_u_to_seqvector

    def eval_(self, clients):
        self.model.eval()
        with torch.no_grad():
            return clients.eval_(self.model.state_dict())



