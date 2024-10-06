# FELLAS
This is the draft code for "FELLAS: Privacy-preserving Federated Sequential Recommendation with LLM as External Service".

# File description
1. clients.py: this file includes two classes: ClientDataManager is used to manage/convert the client's data to training form, and FedRecClients plays as the client in FedSeqRec, including actions like training local model, query for sequence-level service, sequence perturbation, and so on.
2. data.py: this file is responsible for loading data.
3. evaluate.py: this file includes the metrics calculation functions.
4. InfoNCE.py: the contrastive loss function.
5. llmserver.py: this is the LLM server that implement item and sequence embedding functions.
6. main.py: this file is responsible for initialization and running FELLAS.
7. models.py: sequential recommendation model.
8. parse.py: it includes all hyper-parameters.
9. server.py: it is the central server, including coordinating clients for training, querying for item embedding service, evaluation, and so on.

# How to run
1. Prepare data (or Use the processed data in the Data directory)
    - Download data from [Amazon dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
    - Remove items without meta information. After that, remove users that have less than 3 interactions.
    - Rerank the interaction according to the timestamp and leave the last two items as valid and test data.
3. Download LLM models from [HuggingFace](https://huggingface.co/models). The following are the LLMs that we have explored in Table 5.
    - [BERT](https://huggingface.co/google-bert/bert-base-cased/tree/main)
    - [Longformer](https://huggingface.co/allenai/longformer-base-4096/tree/main)
    - [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main)
    - [Llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main)
5. Adjust the hyper-parameters in parse.py. 
6. Run with python main.py.
