# -*- coding: utf-8 -*-
import os
import pdb
from collections import defaultdict

import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from prefetch_generator import BackgroundGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

from utils.dataset import CustomDataSet
from utils.ema import EMA
from utils.tools import get_optimizer_and_scheduler, setup_seed
from models.DTIModel import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def getConfig():
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default="C_elegans")

    """ train args """
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--Batch_size', type=int, default=64)
    parser.add_argument('--Epoch', type=int, default=60)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--early_stop_nums', type=int, default=10)

    parser.add_argument('--drug_encoder_learning_rate', type=float, default=2e-4)
    parser.add_argument('--protein_encoder_learning_rate', type=float, default=2e-4)
    parser.add_argument('--interaction_learning_rate', type=float, default=1e-4)

    """ Protein Model Parameters """
    parser.add_argument('--protein_max_length', type=int, default=2000, help='氨基酸序列长度')
    parser.add_argument('--protein_hidden_size', type=int, default=128)
    parser.add_argument('--protein_conv', type=int, default=40)
    parser.add_argument('--n_gram', type=int, default=3)

    """ Drug Model Parameters """
    parser.add_argument('--hidden_size', type=int, default=128, help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False, help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=4, help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ELU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=True,
                        help='Undirected edges (always sum the two relevant bond vectors)')

    return parser.parse_args()


def evalueate(model, dataset_load, LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []

    with torch.no_grad():
        for batch in dataset_load:
            '''data preparation '''
            inputs = {
                "mol_graph": batch["mol_graph"],
                "smiles": batch["smiles"].to(args.device),
                "smiles_mask": batch["smiles_mask"].to(args.device),
                "protein": batch["protein"].to(args.device),
                "protein_mask": batch["protein_mask"].to(args.device),
            }
            labels = batch["label"].to(args.device)
            predicted_scores = model(**inputs)

            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()

            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())

    try:
        Precision = precision_score(Y, P, zero_division=1)
        Recall = recall_score(Y, P)
        AUC = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC = auc(fpr, tpr)
        Accuracy = accuracy_score(Y, P)
        test_loss = np.average(test_losses)
    except:
        pdb.set_trace()

    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]]
             for i in range(len(sequence) - ngram + 1)]
    return words


if __name__ == "__main__":
    """init hyperparameters"""
    args = getConfig()

    """select seed"""
    setup_seed(args)

    """ init logging """
    os.makedirs(f'./log/{args.dataset}', exist_ok=True)
    logging.basicConfig(
        filename=f'./log/{args.dataset}/{args.dataset}_seed{args.seed}_dim128.log', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w'
    )
    logging.info(args.__dict__)

    """ load data """
    with open(f"./data/{args.dataset}/celegans_data.txt") as f:
        data_list = [x.strip().split(" ") for x in f.readlines()]

    data_list = shuffle_dataset(data_list, args.seed)

    """ initialize n-gram word_dict """
    word_dict = defaultdict(lambda: len(word_dict))
    processed_data_list = []
    for line in data_list:
        smiles, sequence, label = line
        words = split_sequence(sequence, args.n_gram)
        processed_data_list.append([smiles, words, label])

    logging.info(f"n_words: {len(word_dict)}")

    skf = StratifiedKFold(n_splits=5)
    labels = [int(x[-1]) for x in processed_data_list]

    results = np.array([0.0] * 4)
    for fold, (train_idx, test_idx) in enumerate(skf.split(processed_data_list, labels)):

        logging.info(f"start {fold + 1}-th Fold training")

        train_data_list = [processed_data_list[idx] for idx in train_idx]
        test_data_list = [processed_data_list[idx] for idx in test_idx]
        valid_data_list, test_data_list = split_dataset(test_data_list, 0.5)

        """ load train/test data """
        train_dataset = CustomDataSet(args, train_data_list)
        valid_dataset = CustomDataSet(args, valid_data_list)
        test_dataset = CustomDataSet(args, test_data_list)

        train_dataset_load = DataLoader(
            train_dataset, batch_size=args.Batch_size, shuffle=True, num_workers=0,
            collate_fn=CustomDataSet.collate_fn_ngram
        )
        valid_dataset_load = DataLoader(
            valid_dataset, batch_size=args.Batch_size, shuffle=False, num_workers=0,
            collate_fn=CustomDataSet.collate_fn_ngram
        )
        test_dataset_load = DataLoader(
            test_dataset, batch_size=args.Batch_size, shuffle=False, num_workers=0,
            collate_fn=CustomDataSet.collate_fn_ngram
        )

        """create model"""
        model = Model(args, len(word_dict))

        """weight initialize"""
        num_total_steps = len(train_dataset_load) * int(args.Epoch)
        iters = len(train_dataset_load)
        optimizer, scheduler = get_optimizer_and_scheduler(args, model, num_total_steps)
        model = model.to(args.device)

        """load trained model"""
        Loss = nn.CrossEntropyLoss(weight=None)

        """Start training."""
        best_auc, best_aupr = 0.0, 0.0

        step = 0
        early_stop = 0

        ema = EMA(model, decay=args.ema_decay)
        ema.register()
        for epoch in range(1, args.Epoch + 1):
            torch.cuda.empty_cache()

            trian_pbar = tqdm(
                BackgroundGenerator(train_dataset_load),
                total=len(train_dataset_load), desc=f"[Epoch: {epoch}]")
            """train"""
            train_losses_in_epoch = []
            model.train()
            for i, batch in enumerate(trian_pbar):
                step += 1

                optimizer.zero_grad()
                '''data preparation '''
                inputs = {
                    "mol_graph": batch["mol_graph"],
                    "smiles": batch["smiles"].to(args.device),
                    "smiles_mask": batch["smiles_mask"].to(args.device),
                    "protein": batch["protein"].to(args.device),
                    "protein_mask": batch["protein_mask"].to(args.device),
                }
                trian_labels = batch["label"].to(args.device)
                predicted_interaction = model(**inputs)
                train_loss = Loss(predicted_interaction, trian_labels)

                train_losses_in_epoch.append(train_loss.item())  # 将每个batch的loss加入列表

                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
                optimizer.step()
                # scheduler.step(epoch + i / iters)
                scheduler.step()

                ema.update()

                trian_pbar.set_postfix(loss=np.average(train_losses_in_epoch))
                # trian_pbar.set_postfix(loss=train_loss.item())

            """ Test """

            ema.apply_shadow()

            _, _, _, Accuracy, Precision, Recall, AUC, PRC = evalueate(model, valid_dataset_load, Loss)
            # _, _, _, Accuracy, Precision, Recall, AUC, PRC = evalueate(model, test_dataset_load, Loss)

            if best_auc <= AUC or best_aupr <= PRC:
                best_auc = AUC
                best_aupr = PRC
                torch.save(model.state_dict(), f"./checkpoints/{args.dataset}.bin")
                early_stop = 0
            else:
                early_stop += 1

            logging.info(
                f'Epoch: {epoch} ' +
                f'valid_AUC: {AUC:.5f} ' +
                f'valid_PRC: {PRC:.5f} ' +
                f'valid_Accuracy: {Accuracy:.5f} ' +
                f'valid_Precision: {Precision:.5f} ' +
                f'valid_Recall: {Recall:.5f} ' +
                f'best_AUC: {best_auc: .5f} ' +
                f'best_AUPR: {best_aupr: .5f} '
            )

            ema.restore()

            if early_stop >= args.early_stop_nums:
                break

        """ Test """
        model = Model(args, len(word_dict))
        model.load_state_dict(torch.load(f"./checkpoints/{args.dataset}.bin"))
        model.to(args.device)

        _, _, _, Accuracy, Precision, Recall, AUC, PRC = evalueate(model, test_dataset_load, Loss)
        results += np.array([AUC, PRC, Precision, Recall])

        logging.info(
            f'Fold: {fold} ' +
            f'test_AUC: {AUC:.5f} ' +
            f'test_PRC: {PRC:.5f} ' +
            f'test_Accuracy: {Accuracy:.5f} ' +
            f'test_Precision: {Precision:.5f} ' +
            f'test_Recall: {Recall:.5f} '
        )

    results /= 5
    logging.info('\t'.join(map(str, results)) + '\n')
