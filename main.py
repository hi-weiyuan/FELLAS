import copy

import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset
from clients import FedRecClients
from server import FedRecServer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    t0 = time()

    train_data, val_data, test_data, item_meta_data, item2id, id2item = load_dataset(args)

    m_item = max(id2item.keys()) + 1
    args.num_items = m_item

    server = FedRecServer(args, item2id, item_meta_data)

    clients = FedRecClients(args, train_data, val_data, test_data, m_item, server.word_item_embed)

    print("Load data done [%.1f s]. #user=%d, #item=%d" % (time() - t0, len(clients), m_item))
    print("output format: ({Recall@10, NDCG@10}), {Recall@20, NDCG@20})")

    t1 = time()

    server_result = server.eval_(clients)
    print("Iteration 0(init), (%.7f, %.7f) at Rank10, (%.7f, %.7f) at Rank20, " % tuple(server_result) +
          " [%.1fs]" % (time() - t1))

    for epoch in range(1, args.epochs + 1):
        t1 = time()
        rand_clients = copy.deepcopy(clients.get_all_client_ids())
        np.random.shuffle(rand_clients)

        total_loss = []
        llm_u_to_seqvector, neg_llm_u_to_seqvector = server.call_llm_help(clients)

        for i in range(0, len(rand_clients), args.batch_size):
            batch_clients_idx = rand_clients[i: i + args.batch_size]
            loss = server.train_(clients, batch_clients_idx, llm_u_to_seqvector, neg_llm_u_to_seqvector)
            total_loss.append(loss)

        t2 = time()

        server_result = server.eval_(clients)
        print("Iteration %d, client loss = %.5f [%.1fs]" % (epoch,  sum(total_loss) / len(total_loss), t2 - t1) +
              ", (%.7f, %.7f) at Rank10, (%.7f, %.7f) at Rank20" % tuple(server_result) +
              " [%.1fs]" % (time() - t2))

if __name__ == "__main__":
    main()
