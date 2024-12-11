import torch
from torch.utils.data import Dataset
import pandas as pd

class StockDataset(Dataset):
    def __init__(self, csv_file, sequence_len=None):
        """
        csv_file: Path to the CSV file that contains a 'buckets' column.
        sequence_len: Optional length of sequences to chunk data into. If None, 
                      returns the full sequence as a single sample.
        """
        self.df = pd.read_csv(csv_file)
        
        # Extract the buckets column
        self.tokens = self.df['buckets'].astype(str).tolist()
        
        # Build a vocabulary of unique tokens
        self.vocab = sorted(list(set(self.tokens)))
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        
        # Convert tokens to IDs
        self.ids = [self.token2id[t] for t in self.tokens]
        
        # Create input_ids and labels by shifting
        # For language modeling: next token prediction
        # input_ids: [x0, x1, x2, ..., x_(N-2), x_(N-1)]
        # labels:   [x1, x2, x3, ..., x_(N-1),    ?  ]
        # The last token doesn't have a "next" token, you can drop it.
        
        self.input_ids = self.ids[:-1]
        self.labels = self.ids[1:]
        
        # If sequence_len is provided, we will split the data into chunks of sequence_len
        # Each chunk: input_ids[i:i+sequence_len], labels[i:i+sequence_len]
        # If not divisible, the last chunk might be shorter.
        self.sequence_len = sequence_len
        if sequence_len is not None:
            # Number of sequences we can form
            self.num_sequences = len(self.input_ids) // self.sequence_len
        else:
            self.num_sequences = 1  # The entire sequence is one sample

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self.sequence_len is not None:
            start = idx * self.sequence_len
            end = start + self.sequence_len
            input_ids = self.input_ids[start:end]
            labels = self.labels[start:end]
        else:
            # Return the entire sequence as one item
            input_ids = self.input_ids
            labels = self.labels

        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {"input_ids": input_ids_tensor, "labels": labels_tensor}

# Example usage:
dataset = StockDataset("data/bucket_data.csv", sequence_len=128)
# If no sequence_len is given, it will treat the entire sequence as one sample.
item = dataset[0]
print(item["input_ids"], item["labels"])
