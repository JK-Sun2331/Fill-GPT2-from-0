import json
import tiktoken
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self,path,max_seq_len):

        self.enc = tiktoken.get_encoding("gpt2")
        