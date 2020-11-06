import torch
from torch.utils.data import Dataset

num_tag = 12
class MovieCommentDataset(Dataset):
    def __init__(self, data, max_len, padding_idx):
        super(MovieCommentDataset, self).__init__()
        self.data_size = len(data["text"])
        self.text = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx
        self.category = torch.ones((self.data_size, num_tag), dtype=torch.int64) * padding_idx

        text_length = list()
        tag_length = list()

        for idx, content in enumerate(data["text"]):
            content_len = min(len(content), max_len)
            tag_len = min(len(data["class"][idx]), 12)
            self.text[idx][:content_len] = torch.tensor(data["text"][idx][:content_len], dtype=torch.int64)
            text_length.append(content_len)
            self.category[idx][:tag_len] = torch.tensor(data["class"][idx][:tag_len], dtype=torch.int64)
            tag_length.append(tag_len)

        self.text_length = torch.tensor(text_length)
        self.tag_length = torch.tensor(tag_length)


    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        ret_data = {
            "text": self.text[idx],
            "text_length": self.text_length[idx],
            "tag": self.category[idx],
            "tag_length": self.tag_length[idx]
        }
        return ret_data
