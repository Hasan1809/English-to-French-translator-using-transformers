import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

def encode_with_special_tokens(sentence, sp_model, add_bos=True, add_eos=True, max_length=512):

    tokens = sp_model.encode(sentence)
    
    if add_bos:
        tokens = [2] + tokens  
    if add_eos:
        tokens = tokens + [3]  
    
    if len(tokens) > max_length:
        if add_eos:
            tokens = tokens[:max_length-1] + [3] 
        else:
            tokens = tokens[:max_length]
    
    return torch.tensor(tokens, dtype=torch.long)

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, sp_model, max_length=512):
        self.sp_model = sp_model
        self.max_length = max_length
        
        self.src = []
        self.tgt = []
        
        for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
            
            src_tokens = encode_with_special_tokens(
                src_sent, sp_model, add_bos=False, add_eos=True, max_length=max_length
            )
            
            
            tgt_tokens = encode_with_special_tokens(
                tgt_sent, sp_model, add_bos=True, add_eos=True, max_length=max_length
            )
            
            if len(src_tokens) > 1 and len(tgt_tokens) > 2:  
                self.src.append(src_tokens)
                self.tgt.append(tgt_tokens)
        
        print(f"Dataset created with {len(self.src)} sentence pairs")
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def collate_fn(batch):

    src_batch, tgt_batch = zip(*batch)
    
    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=False)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0, batch_first=False)
    

    src_padded = src_padded.transpose(0, 1) 
    tgt_padded = tgt_padded.transpose(0, 1)  
    
    return src_padded, tgt_padded

def create_dataloader(english_sentences, french_sentences, sp_model, batch_size=16, 
                     max_length=512, shuffle=True, num_workers=0):
    dataset = TranslationDataset(
        english_sentences, 
        french_sentences, 
        sp_model, 
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader

# Example usage and testing
def test_dataloader(dataloader, sp_model, num_batches=2):
    
    for i, (src_batch, tgt_batch) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        print(f"\nBatch {i+1}:")
        print(f"Source batch shape: {src_batch.shape}")  
        print(f"Target batch shape: {tgt_batch.shape}")  
        
        print(f"\nFirst example:")
        src_tokens = src_batch[0].tolist()
        tgt_tokens = tgt_batch[0].tolist()
        
        print(f"Source tokens: {src_tokens[:20]}...")  # First 20 tokens
        print(f"Target tokens: {tgt_tokens[:20]}...")
        
        src_text = sp_model.decode([t for t in src_tokens if t != 0])  # Remove padding
        tgt_text = sp_model.decode([t for t in tgt_tokens if t != 0])
        
        print(f"Source text: {src_text}")
        print(f"Target text: {tgt_text}")
        







