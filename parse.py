import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run FELLAS.")
    parser.add_argument('--device_id', nargs='?', default='cuda:3' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')
    parser.add_argument('--data_path', nargs='?', default='/root/Data/PLG')

    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    parser.add_argument('--epochs', type=int, default=20, help='Number of full global epochs.')
    parser.add_argument('--local_epoch', type=int, default=5, help='Number of local training epochs.')
    parser.add_argument('--neg_num', type=int, default=1, help='Number of negative items. 1 or 4?')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size of clients.')
    parser.add_argument('--max_len', type=int, default=50, help='max length of a sequence.')
    parser.add_argument('--lr', type=float, default=0.001, help='global Learning rate. 0.01 or 0.001')

    parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate.')
    parser.add_argument('--embed_dim', type=int, default=8, help='Dim of server latent vectors.')
    parser.add_argument('--n_blocks', type=int, default=2, help='Number of client_n_blocks.')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of client_n_heads.')

    parser.add_argument('--cl_loss_factor', type=float, default=1., help='control the strength of contrastive learning.')
    parser.add_argument('--std', type=float, default=0.01, help='private parameter.')
    parser.add_argument('--original_pretrain_ckpt', type=str, default="/root/longformer-base-4096/")
    return parser.parse_args()


args = parse_args()
