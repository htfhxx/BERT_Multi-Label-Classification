import argparse
import json
import numpy as np
import pickle

from transformers import BertTokenizer

all_tag_list = {'怀旧': 1, '清新': 2, '浪漫': 3, '伤感': 4, '治愈': 5, '放松': 6, '孤独': 7, '感动': 8, '兴奋': 9, '快乐': 10, '安静': 11, '思念': 12}

def process_set(file, tokenizer, nlu_source):
    content = {
        "text": list(),
        "class": list()
    }
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            #text_content = sample['total_comments'][0].strip()
            if nlu_source == 'comment':
                text_content = sample['comment'][0]
            elif nlu_source == 'content':
                text_content = sample['source']
            elif nlu_source == 'comment_content':
                text_content = sample['comment'] + '；' + sample['source']
            elif nlu_source == 'content_comment':
                text_content =  sample['source'] + '；'  + sample['comment']


            tag_list = sample['tag_list']
            # print('tag_list:', tag_list)

            content["class"].append([int(all_tag_list[tag]) for tag in tag_list])
            text_idx = tokenizer.encode(text_content)
            content["text"].append(text_idx)
        # print('content["class"]:', content["class"])
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/preprocess_data_bert.config")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = json.loads(config_file.read())
    tokenizer = BertTokenizer.from_pretrained("./bert-base-chinese/vocab.txt")

    nlu_source_choices = ["comment", "content", "comment_content", 'content_comment']

    for nlu_source in nlu_source_choices:
        indexed_train = process_set(config["train_dir"], tokenizer, nlu_source)
        indexed_dev = process_set(config["dev_dir"], tokenizer, nlu_source)
        indexed_test = process_set(config["test_dir"], tokenizer, nlu_source)
        with open(config["indexed_data_dir"]+'/'+str(nlu_source)+'/indexed_train_bert.json', "w") as f:
            f.write(json.dumps(indexed_train))
        with open(config["indexed_data_dir"]+'/'+str(nlu_source)+'/indexed_dev_bert.json', "w") as f:
            f.write(json.dumps(indexed_dev))
        with open(config["indexed_data_dir"]+'/'+str(nlu_source)+'/indexed_test_bert.json', "w") as f:
            f.write(json.dumps(indexed_test))
        print('finished '+ nlu_source)


if __name__ == "__main__":
    main()
