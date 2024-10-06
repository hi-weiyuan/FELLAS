import copy
from tqdm import tqdm
from models import *
from parse import args
import torch
import torch.nn as nn
import numpy as np
from evaluate import evaluate_recall, evaluate_ndcg, evaluate_hr


class ClientDataManager:
    def __init__(self, train_data, val_data, test_data, all_full_client_ids, m_item, args):
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self.all_full_client_ids = all_full_client_ids
        self.m_item = m_item
        self.args = args
        self._train_ = {k: v + self._val_data[k] for k, v in self._train_data.items() if k in all_full_client_ids}
        self.build_data()

    def build_data(self):
        self.all_input_seq = {}
        self.all_val_seq = {}
        self.all_seq_len = {}
        self.all_neg_seq = {}
        self.all_pos_seq = {}
        for k, train_ind in self._train_.items():
            input_seq, valid_seq, seq_len, neg_seq, pos = self.__construct_client_local_dataset(train_ind, self.args.max_len)
            self.all_input_seq[k] = torch.LongTensor(input_seq).unsqueeze(0)
            self.all_val_seq[k] = torch.LongTensor(valid_seq).unsqueeze(0)
            self.all_seq_len[k] = torch.tensor(seq_len).unsqueeze(0)
            self.all_neg_seq[k] = torch.from_numpy(neg_seq).unsqueeze(0)
            self.all_pos_seq[k] = torch.LongTensor(pos).unsqueeze(0)

    def build_neg_data_for_query(self):
        tmp_all_neg_seq = {}
        for k, train_ind in self._train_.items():
            _, _, _, neg_seq, _ = self.__construct_client_local_dataset(train_ind, self.args.max_len)
            tmp = torch.from_numpy(neg_seq).squeeze().tolist()
            if type(tmp) == int:
                tmp = [tmp]
            tmp_all_neg_seq[k] = tmp
        return tmp_all_neg_seq

    def __construct_client_local_dataset(self, train_ind, max_len):
        start = len(train_ind) > max_len and -max_len or 0
        end = len(train_ind) > max_len and max_len - 1 or len(train_ind) - 1
        input_seq = train_ind[start:-1]
        valid_seq = train_ind[start + 1:]

        seq_len = end
        pos = list(range(1, end + 1))
        all_items = [i for i in range(0, self.m_item)]
        cand = np.setdiff1d(np.array(all_items), np.array(train_ind))
        neg_seq = np.random.choice(cand, (seq_len, self.args.neg_num))
        return input_seq, valid_seq, seq_len, neg_seq, pos

    def __getitem__(self, user_id):
        input_seq = self.all_input_seq[user_id]
        val_seq = self.all_val_seq[user_id]
        seq_len = self.all_seq_len[user_id]
        neg_seq = self.all_neg_seq[user_id]
        pos = self.all_pos_seq[user_id]
        return input_seq, val_seq, seq_len, neg_seq, pos

    def get_seq_for_prediction(self, user_id):
        return self._train_[user_id][-self.args.max_len:]

    def __len__(self):
        return len(self._train_)
    def get_max_item_id(self):
        return self.m_item
    def get_user_set(self):
        return self.all_full_client_ids



class FedRecClients(nn.Module):
    def __init__(self, config, train_data, val_data, test_data, m_item, word_item_emb):
        super().__init__()
        self.model = Model(config, word_item_emb)
        self.config = config
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self.m_item = m_item

        train_client_ids = list(self._train_data.keys())
        val_client_ids = list(self._val_data.keys())
        test_client_ids = list(self._test_data.keys())
        self.all_full_client_ids = list(set(train_client_ids).intersection(val_client_ids, test_client_ids))
        self.data_manager = ClientDataManager(train_data, val_data, test_data, self.all_full_client_ids, m_item, config)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = torch.nn.BCELoss()
        self.codebook = {}

    def __len__(self):
        return len(self.all_full_client_ids)
    def get_all_client_ids(self):
        return self.all_full_client_ids

    def train_single_batch(self, inp_seq, target_seq, neg_seq, pos_seq,
                           u_presentation, u_neg_presentation):
        inp_seq, target_seq, neg_seq, pos_seq = inp_seq.to(self.config.device_id), target_seq.to(self.config.device_id), \
                                                neg_seq.to(self.config.device_id), pos_seq.to(self.config.device_id)

        u_presentation = u_presentation.to(self.config.device_id)
        u_neg_presentation = u_neg_presentation.to(self.config.device_id)

        seq_out = self.model(inp_seq, pos_seq)
        loss = self.model.loss_function(seq_out, target_seq, neg_seq, pos_seq)
        cl_loss = self.model.u_contrastive_loss(seq_out, u_presentation, u_neg_presentation)
        loss = loss + cl_loss * self.config.cl_loss_factor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_(self, uids, received_model_dict, llm_u_vector, llm_u_neg_vector):
        epoch_loss = []
        client_grads = {}

        for u_id in uids:
            self.model.load_state_dict(received_model_dict)
            self.model = self.model.to(self.config.device_id)
            self.model.train()
            input_seq, val_seq, seq_len, neg_seq, pos = self.data_manager[u_id]

            u_presentation = llm_u_vector[u_id]
            u_neg_presentation = llm_u_neg_vector[u_id]
            for _ in range(self.config.local_epoch):
                loss = self.train_single_batch(input_seq, val_seq, neg_seq, pos, u_presentation, u_neg_presentation)
                epoch_loss.append(loss)
            self.model = self.model.cpu()
            grad_u = {}
            current_model_state = copy.deepcopy(self.model.state_dict())
            for k, v in current_model_state.items():
                grad_u[k] = (current_model_state[k] - received_model_dict[k]).detach().clone()
            client_grads[u_id] = grad_u

        return client_grads, sum(epoch_loss) / len(epoch_loss)

    def noise(self, shape, std):
        noise = np.random.multivariate_normal(
            mean=np.zeros(shape[1]), cov=np.eye(shape[1]) * std, size=shape[0]
        )
        return torch.Tensor(noise).to(args.device_id)

    def query(self):
        perturbed_train_data, perturbed_val_data, perturbed_test_data = self.d_privacy_preserving()
        return perturbed_train_data, perturbed_val_data, perturbed_test_data

    def query_for_neg(self):
        neg_data = self.data_manager.build_neg_data_for_query()
        return neg_data

    def generate_codebook(self):
        item_emb = self.model.word_item_emb.weight.clone().detach().requires_grad_(False)
        noise = self.noise(item_emb.shape, std=self.config.std)
        item_emb = item_emb + noise
        for i in range(item_emb.size(0)):
            similarities = nn.functional.cosine_similarity(item_emb[i], item_emb, dim=-1)
            similarities[i] -= 2
            _, indexes = torch.topk(similarities, k=1)
            indexes = indexes.squeeze().cpu().tolist()
            self.codebook[i] = indexes

    def d_privacy_preserving(self):
        self.generate_codebook()
        perturbed_train_data = {}
        perturbed_val_data = {}
        perturbed_test_data = {}
        for k, v in self._train_data.items():
            perturbed_train_data[k] = [self.codebook[vv] for vv in v]
        for k, v in self._val_data.items():
            perturbed_val_data[k] = [self.codebook[vv] for vv in v]
        for k, v in self._test_data.items():
            perturbed_test_data[k] = [self.codebook[vv] for vv in v]
        return perturbed_train_data, perturbed_val_data, perturbed_test_data

    def get_client_prediction(self, uid):
        inp_seq = self.data_manager.get_seq_for_prediction(uid)

        inp_len = len(inp_seq)
        pos_seq = list(range(1, inp_len + 1))
        inp_seq = torch.LongTensor(inp_seq).unsqueeze(0).to(self.config.device_id)
        input_len = torch.tensor(inp_len).unsqueeze(0)
        pos_seq = torch.LongTensor(pos_seq).unsqueeze(0).to(self.config.device_id)
        seq_emb = self.model(inp_seq, pos_seq)
        last_item_embeddings = seq_emb[:, input_len[0] - 1, :]
        predictions = torch.matmul(last_item_embeddings, self.model.item_emb.transpose(0, 1))
        return predictions.squeeze()

    def eval_(self, received_model_state):
        self.model.load_state_dict(received_model_state)
        self.model = self.model.to(self.config.device_id)
        self.model.eval()

        test_cnt, test_results = 0, 0.
        with torch.no_grad():
            for uid in self.all_full_client_ids:
                prediction = self.get_client_prediction(uid)
                prediction[self.data_manager._train_[uid]] = - (1 << 10)
                golden_label = self.data_manager._test_data[uid]
                hr_at_10 = evaluate_recall(prediction, golden_label, 10)
                ndcg_at_10 = evaluate_ndcg(prediction, golden_label, 10)
                hr_at_20 = evaluate_recall(prediction, golden_label, 20)
                ndcg_at_20 = evaluate_ndcg(prediction, golden_label, 20)
                current_test_result = np.array([hr_at_10, ndcg_at_10, hr_at_20, ndcg_at_20])
                test_results += current_test_result
                test_cnt += 1

        return test_results / test_cnt


