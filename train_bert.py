import argparse
import json
import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from Data import MovieCommentDataset
from bert_model import BertClassifier
from sklearn.metrics import f1_score
import numpy as np

num_tag =12


def micro_f1(y_true, y_pred):
    # print('y_true: ', y_true)
    # print('y_pred: ', y_pred)
    tp = ((y_true==1) & (y_pred==True)).sum()
    fp = ((y_true==0) & (y_pred==True)).sum()
    fn = ((y_true==1) & (y_pred==False)).sum()
    if tp == 0:
        return 0.0, 0.0, 0.0
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    return precision, recall, 2 * (precision * recall) / (precision + recall)


def F1(labels, prediction):
    precisions, recalls, F1s = [], [], []
    for label, pred in zip(labels, prediction):
        precision, recall, F1  = micro_f1(label, pred)
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
    return precisions, recalls, F1s


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')

    def set_scorer(self, scorer):
        self.scorer = scorer


    def get_scores(self, name, logits):
        size = logits.size(0)
        ret = torch.tensor(getattr(self.scorer, name)[-size:]).float()
        if len(ret.size()) == 2:
            ret = ret.mean(dim=-1)
        return ret


    def nlu_loss(self, logits, targets):
        loss = [
            self.BCE(logits[:, i], targets).mean(-1)
            for i in range(logits.size(1))
        ]
        return torch.stack(loss, dim=0).transpose(0, 1)

    def nlu_score(self, decisions, targets, average):

        device = decisions.device
        decisions = decisions.detach().cpu().long().numpy()
        targets = targets.detach().cpu().long().numpy()
        scores = [
            [
                f1_score(y_true=np.array([label]), y_pred=np.array([pred]), average=average)
                for label, pred in zip(targets, decisions[:, i])
            ]
            for i in range(decisions.shape[1])
        ]
        return torch.tensor(scores, dtype=torch.float, device=device).transpose(0, 1)


    def forward(self, logits, targets):
        # print('logits.shape: ',logits.shape )
        # print('logits: ',logits )
        # print('targets.shape: ', targets.shape)
        # print('targets: ', targets)
        # exit()
        logits = logits.contiguous()
        targets = targets.contiguous()
        sup_loss  = 0
        splits = logits.split(split_size=1, dim=1)
        sup_loss += self.BCE(logits.squeeze(1), targets).mean()
        return sup_loss




class MultilabelScorer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.F1s = []
        self.precisions = []
        self.recalls = []



    def update(self, labels, prediction):
        # print('labels: ', labels)
        # print('prediction: ', prediction)
        precisions, recalls, F1s = F1(labels, prediction)
        # print('micro_f1: ',micro_f1)
        self.F1s.extend(precisions)
        self.precisions.extend(recalls)
        self.recalls.extend(F1s)


    def get_avg_scores(self):
        avg_F1 = np.mean(self.F1s)
        avg_precisions = np.mean(self.precisions)
        avg_recalls = np.mean(self.recalls)

        return avg_precisions,  avg_recalls, avg_F1

    def print_avg_scores(self):
        avg_precisions,  avg_recalls, avg_F1 = self.get_avg_scores()
        print(f"Average micro precisions: {avg_precisions}")
        print(f"Average micro recalls: {avg_recalls}")
        print(f"Average micro f1: {avg_F1}")




def sequences_to_nhot(seqs, attr_vocab_size):
    labels = np.zeros((len(seqs), attr_vocab_size), dtype=np.int)

    for bid, seq in enumerate(seqs):
        for word in seq:
            if word>0:
                labels[bid][word-1] = 1
    # print('seqs: ', seqs)
    # print('labels: ', labels)
    return labels

def train_epoch(device, model, criterion, optimizer, train_loader):
    model.train()

    train_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, ncols=80)
    scorer = MultilabelScorer()
    for idx, batch in enumerate(train_loader_tqdm):
        # if idx>10:
        #     break
        text = batch["text"].to(device)
        tag_list = batch["tag"]
        optimizer.zero_grad()
        logits = model(text)
        # print('logits: ', logits)
        tag_list = tag_list.clone().numpy()
        targets = sequences_to_nhot(tag_list, num_tag)
        targets = torch.from_numpy(targets).float()
        #print('targets: ',targets)



        prediction = (torch.sigmoid(logits.detach().cpu()) >= 0.5)


        prediction = prediction.clone().numpy()
        #print('prediction2: ',prediction)

        targets_clone = targets.detach().cpu().long().numpy()
        scorer.update(targets_clone, prediction)

        loss = criterion(logits.cpu().unsqueeze(1), targets.cpu(),)


        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        description = "Avg. batch train loss: {}".format(train_loss / (idx + 1))
        train_loader_tqdm.set_description(description)
    train_loss /= len(train_loader)
    avg_precisions,  avg_recalls, avg_F1 = scorer.get_avg_scores()
    avg_precisions = '{:.4f}'.format(avg_precisions)
    avg_recalls = '{:.4f}'.format(avg_recalls)
    avg_F1 = '{:.4f}'.format(avg_F1)
    train_loss = '{:.4f}'.format(train_loss)
    return train_loss, avg_precisions,avg_recalls, avg_F1


def dev_epoch(device, model, criterion, dev_loader):
    model.eval()

    dev_loss = 0.0
    dev_loader_tqdm = tqdm(dev_loader, ncols=80)
    scorer = MultilabelScorer()
    for idx, batch in enumerate(dev_loader_tqdm):
        # if idx>10:
        #     break
        text = batch["text"].to(device)
        tag_list = batch["tag"]

        logits = model(text)
        tag_list = tag_list.clone().numpy()
        targets = sequences_to_nhot(tag_list, num_tag)
        # print('tag_list: ', tag_list)
        # print('targets: ', targets)
        targets = torch.from_numpy(targets).float()

        prediction = (torch.sigmoid(logits.detach().cpu()) >= 0.5)
        prediction = prediction.clone().numpy()

        targets_clone = targets.detach().cpu().long().numpy()
        scorer.update(targets_clone, prediction)
        loss = criterion(logits.cpu().unsqueeze(1), targets.cpu(),)

        dev_loss += loss.item()
        description = "Avg. batch dev loss: {}".format(dev_loss / (idx + 1))
        dev_loader_tqdm.set_description(description)
    dev_loss /= len(dev_loader_tqdm)
    avg_precisions,  avg_recalls, avg_F1 = scorer.get_avg_scores()
    avg_precisions = '{:.4f}'.format(avg_precisions)
    avg_recalls = '{:.4f}'.format(avg_recalls)
    avg_F1 = '{:.4f}'.format(avg_F1)
    dev_loss = '{:.4f}'.format(dev_loss)

    return dev_loss, avg_precisions, avg_recalls, avg_F1


def train(model, device, train_loader, dev_loader, test_loader, config):

    checkpoint_dir = config["checkpoint_dir"],
    epoches = config["epoches"]
    learning_rate = config["learning_rate"]
    max_patience_epoches = config["max_patience_epoches"]
    criterion = Criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    patience_count, best_f1 = 0, 0
    start_epoch = 0
    for epoch in range(start_epoch, epoches):
        train_loss,  avg_precisions,avg_recalls, avg_F1 = train_epoch(
            device=device,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader
        )
        print("* Train epoch {}".format(epoch + 1))
        print("-> loss={}, precisions={}, recalls={}, f1={}".format(train_loss, avg_precisions, avg_recalls, avg_F1))

        with open('log/train_result.csv', 'a') as file:
            file.write("{},{},{},{},{}\n".format(epoch, train_loss,avg_precisions, avg_recalls, avg_F1))

        dev_loss, avg_precisions, avg_recalls, dev_f1 = dev_epoch(
            device=device,
            model=model,
            criterion=criterion,
            dev_loader=dev_loader
        )
        print("* Dev epoch {}".format(epoch + 1))
        print("-> loss={}, precisions={}, recalls={}, f1={}".format(dev_loss, avg_precisions, avg_recalls, dev_f1))
        with open('log/valid_result.csv', 'a') as file:
            file.write("{},{},{},{},{}\n".format(epoch, dev_loss,avg_precisions, avg_recalls, dev_f1))


        # torch.save({
        #     "model": model.state_dict(),
        #     "optimizer": optimizer.state_dict()
        # }, os.path.join(checkpoint_dir, 'best_bert_model' + str(epoch) + '.tar'))

        if float(dev_f1) >  float(best_f1):
            patience_count = 0
            best_f1 = dev_f1
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(checkpoint_dir, 'best_bert_model.tar'))
            print('new epoch saved as the best model', epoch)

        else:
            patience_count += 1
            if patience_count >= max_patience_epoches:
                print("Early Stopping at epoch {}".format(epoch + 1))
                break

    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_bert_model.tar"))
    model.load_state_dict(checkpoint["model"])
    test_loss, avg_precisions, avg_recalls, test_f1 = dev_epoch(
        device=device,
        model=model,
        criterion=criterion,
        dev_loader=test_loader
    )
    print("* Result on test set")
    print("-> loss={}, f1={}".format(test_loss, test_f1))


def test(model, device, test_loader, checkpoint_dir):



    criterion = Criterion()
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_bert_model.tar"))
    model.load_state_dict(checkpoint["model"])

    test_loss,  avg_precisions, avg_recalls, test_f1  = dev_epoch(
        device=device,
        model=model,
        criterion=criterion,
        dev_loader=test_loader
    )
    print("* Result on test set")
    print("-> loss={}, precisions={}, recalls={}, f1={}".format(test_loss, avg_precisions,avg_recalls, test_f1))

def main():
    '''prepration for config'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="utils/config/train_bert.config")
    parser.add_argument("--mode", default="test")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = json.loads(config_file.read())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 20, "Training on device: {}".format(device), "=" * 20)

    train_dir = config["train_dir"]
    dev_dir = config["dev_dir"]
    test_dir = config["test_dir"]
    max_len = config["max_len"]
    batch_size = config["batch_size"]
    checkpoint_dir = config["checkpoint_dir"]

    # prepration for data
    print("=" * 20, "Preparing training data...", "=" * 20)
    with open(train_dir, "r") as f:
        indexed_train_data = json.loads(f.read())
    train_data = MovieCommentDataset(indexed_train_data, max_len, padding_idx=0)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("=" * 20, "Preparing dev data...", "=" * 20)
    with open(dev_dir, "r") as f:
        indexed_dev_data = json.loads(f.read())
    dev_data = MovieCommentDataset(indexed_dev_data, max_len, padding_idx=0)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    print("=" * 20, "Preparing test data...", "=" * 20)
    with open(test_dir, "r") as f:
        indexed_test_data = json.loads(f.read())
    test_data = MovieCommentDataset(indexed_test_data, max_len, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


    model = BertClassifier(device=device, transformer_width=768).to(device)

    if args.mode is 'train':
        train(model, device, train_loader, dev_loader, test_loader, config)
    elif args.mode is 'test':
        test(model, device, test_loader, checkpoint_dir)


if __name__ == "__main__":
    main()
