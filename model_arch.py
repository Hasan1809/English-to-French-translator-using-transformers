import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TranslationTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, d_model=300, nhead=6, 
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, 
                 dropout=0.1, max_len=512):
        super().__init__()
        
        # Special token IDs
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Changed to True for easier handling
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
    
    def create_padding_mask(self, seq):
        return (seq == self.pad_id)
    
    def create_causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, src, tgt):
        """
        Forward pass for training
        Args:
            src: [batch_size, src_seq_len] - source sequences
            tgt: [batch_size, tgt_seq_len] - target sequences (teacher forcing)
        """
        batch_size, tgt_seq_len = tgt.shape
        device = src.device
        
        # Create masks
        src_padding_mask = self.create_padding_mask(src)  # [batch_size, src_seq_len]
        tgt_padding_mask = self.create_padding_mask(tgt)  # [batch_size, tgt_seq_len]
        tgt_causal_mask = self.create_causal_mask(tgt_seq_len, device)  # [tgt_seq_len, tgt_seq_len]
        
        # Embeddings and positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)  # Scale embeddings
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding (need to transpose for pos_encoder)
        src_emb = src_emb.transpose(0, 1)  # [src_seq_len, batch_size, d_model]
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_seq_len, batch_size, d_model]
        
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Transpose back for batch_first=True
        src_emb = src_emb.transpose(0, 1)  # [batch_size, src_seq_len, d_model]
        tgt_emb = tgt_emb.transpose(0, 1)  # [batch_size, tgt_seq_len, d_model]
        
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Transformer forward pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Final linear layer
        output = self.fc_out(output)  # [batch_size, tgt_seq_len, vocab_size]
        
        return output
    
    def encode(self, src):
        """Encode source sequence"""
        batch_size = src.shape[0]
        device = src.device
        
        # Create source mask
        src_padding_mask = self.create_padding_mask(src)
        
        # Embedding and positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb.transpose(0, 1)
        src_emb = self.pos_encoder(src_emb)
        src_emb = src_emb.transpose(0, 1)
        src_emb = self.dropout(src_emb)
        
        # Encode
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        return memory, src_padding_mask
    
    def decode_step(self, tgt_seq, memory, memory_key_padding_mask):
        """Single decoding step"""
        tgt_seq_len = tgt_seq.shape[1]
        device = tgt_seq.device
        
        # Create masks
        tgt_padding_mask = self.create_padding_mask(tgt_seq)
        tgt_causal_mask = self.create_causal_mask(tgt_seq_len, device)
        
        # Embedding and positional encoding
        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_emb = self.dropout(tgt_emb)
        
        # Decode
        output = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Get logits for the last position
        logits = self.fc_out(output[:, -1:, :])  # [batch_size, 1, vocab_size]
        
        return logits
    
    def translate(self, src_sentence, tokenizer, max_length=100, beam_size=1, temperature=1.0):
        """
        Translate a single sentence
        Args:
            src_sentence: string - source sentence
            tokenizer: SentencePiece tokenizer
            max_length: maximum target sequence length
            beam_size: beam search size (1 = greedy)
            temperature: sampling temperature
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Tokenize input
            src_tokens = tokenizer.encode(src_sentence)
            src_tensor = torch.tensor([src_tokens], device=device)  # [1, src_len]
            
            # Encode
            memory, memory_key_padding_mask = self.encode(src_tensor)
            
            # Initialize target sequence with BOS token
            tgt_seq = torch.tensor([[self.bos_id]], device=device)  # [1, 1]
            
            # Generate tokens one by one
            for _ in range(max_length):
                # Get logits for next token
                logits = self.decode_step(tgt_seq, memory, memory_key_padding_mask)
                logits = logits / temperature
                
                # Sample next token (greedy or sampling)
                if beam_size == 1:
                    next_token = torch.argmax(logits, dim=-1)  # [1, 1]
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs.squeeze(0), 1).unsqueeze(0)  # [1, 1]
                
                # Append to sequence
                tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
                
                # Check for EOS token
                if next_token.item() == self.eos_id:
                    break
            
            # Decode tokens back to text
            output_tokens = tgt_seq.squeeze(0).tolist()[1:]  # Remove BOS token
            if self.eos_id in output_tokens:
                output_tokens = output_tokens[:output_tokens.index(self.eos_id)]  # Remove EOS and after
            
            translated_text = tokenizer.decode(output_tokens)
            return translated_text

# Example usage and initialization
def create_model(embedding_matrix, vocab_size):
    """Create and initialize the translation model"""
    model = TranslationTransformer(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        d_model=300,
        nhead=6,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=64
    )
    
    return model


