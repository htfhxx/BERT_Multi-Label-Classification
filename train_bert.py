import argparse
import json
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.Criterion import sequences_to_nhot, MultilabelScorer, Criterion
from utils.Data import MovieCommentDataset
from model.bert_model import BertClassifier






def train_epoch(config, device, model, criterion, optimizer, train_loader):
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

        tag_list = tag_list.clone().numpy()
        targets = sequences_to_nhot(tag_list, config["num_tag"])
        targets = torch.from_numpy(targets).float()
        targets_clone = targets.detach().cpu().long().numpy()

        prediction = (torch.sigmoid(logits.detach().cpu()) >= 0.5)
        prediction = prediction.clone().numpy()

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


def dev_epoch(config, device, model, criterion, dev_loader):
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
        targets = sequences_to_nhot(tag_list, config["num_tag"])
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


def train(model, train_loader, dev_loader, config):

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
            config=config,
            device=config["device"],
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
            config=config,
            device=config["device"],
            model=model,
            criterion=criterion,
            dev_loader=dev_loader
        )
        print("* Dev epoch {}".format(epoch + 1))
        print("-> loss={}, precisions={}, recalls={}, f1={}".format(dev_loss, avg_precisions, avg_recalls, dev_f1))
        with open('log/valid_result.csv', 'a') as file:
            file.write("{},{},{},{},{}\n".format(epoch, dev_loss,avg_precisions, avg_recalls, dev_f1))


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




def test(model, test_loader, checkpoint_dir, config):
    criterion = Criterion()
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_bert_model.tar"))
    model.load_state_dict(checkpoint["model"])

    test_loss,  avg_precisions, avg_recalls, test_f1  = dev_epoch(
        config=config,
        device=config["device"],
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
    parser.add_argument("--mode", default="train")
    parser.add_argument("--max_len", default=500)
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--epoches", default=30)
    parser.add_argument("--learning_rate", default=0.00001)
    parser.add_argument("--max_patience_epoches", default=10)
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = json.loads(config_file.read())

    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 20, "Running on device: {}".format(config["device"]), "=" * 20)

    config["max_len"] = args.max_len
    config["batch_size"]  = args.batch_size
    config["epoches"] = args.epoches
    config["learning_rate"] = args.learning_rate
    config["max_patience_epoches"] = args.max_patience_epoches



    model = BertClassifier(device=config["device"], transformer_width=768).to(config["device"])

    # prepration for data
    if args.mode is 'train':
        print("=" * 20, "Preparing training data...", "=" * 20)
        with open(config["train_dir"], "r") as f:
            indexed_train_data = json.loads(f.read())
        train_data = MovieCommentDataset(indexed_train_data, config['max_len'], padding_idx=0)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])

        print("=" * 20, "Preparing dev data...", "=" * 20)
        with open(config["dev_dir"], "r") as f:
            indexed_dev_data = json.loads(f.read())
        dev_data = MovieCommentDataset(indexed_dev_data, config['max_len'], padding_idx=0)
        dev_loader = DataLoader(dev_data, shuffle=True, batch_size=config['batch_size'])

        train(model,  train_loader, dev_loader, config)
    elif args.mode is 'test':
        print("=" * 20, "Preparing test data...", "=" * 20)
        with open(config["test_dir"], "r") as f:
            indexed_test_data = json.loads(f.read())
        test_data = MovieCommentDataset(indexed_test_data, config['max_len'], padding_idx=0)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=config['batch_size'])

        test(model, test_loader, config["checkpoint_dir"], config)





if __name__ == "__main__":
    main()
