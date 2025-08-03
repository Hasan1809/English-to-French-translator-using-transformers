import sentencepiece as spm
import torch
import numpy as np
import tempfile
import os
import gensim.downloader as api

def create_proper_sentencepiece_model(english_sentences, french_sentences, model_prefix='translation_bpe'):

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
        temp_path = temp_file.name
        
        # Write all sentences - mix English and French for better shared vocabulary
        total_sentences = 0
        for eng, fr in zip(english_sentences, french_sentences):
            # Clean and write both sentences
            eng_clean = eng.strip()
            fr_clean = fr.strip()
            
            if eng_clean and fr_clean:  # Only write non-empty sentences
                temp_file.write(eng_clean + '\n')
                temp_file.write(fr_clean + '\n')
                total_sentences += 2
    
    print(f"Created training corpus: {temp_path}")
    print(f"Total sentences: {total_sentences}")
    

    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=model_prefix,
        vocab_size=32000,  # Slightly larger for better coverage
        model_type='bpe',
        
        # Critical UTF-8 and character settings
        character_coverage=0.9998,  # Very high coverage for accented chars
        byte_fallback=True,  # Handle any UTF-8 bytes as fallback
        
        # Sentence processing
        input_sentence_size=2000000,  # Process many sentences
        shuffle_input_sentence=True,
        max_sentence_length=512,
        
        # Token IDs
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=['<en>', '<fr>'],  # Language tags
        
        # Text normalization (important for multilingual)
        normalization_rule_name='nmt_nfkc_cf',
        remove_extra_whitespaces=True,
        
        # Advanced settings for better tokenization
        split_by_unicode_script=True,
        split_by_number=True,
        split_by_whitespace=True,
        treat_whitespace_as_suffix=False,
        allow_whitespace_only_pieces=True,
        split_digits=True,
        
        # Training parameters
        num_threads=4,
        train_extremely_large_corpus=False,
        seed_sentencepiece_size=1000000,
        shrinking_factor=0.75,
        num_sub_iterations=2
    )
    
    os.unlink(temp_path)
    
    print(f"\nSentencePiece model created:")
    print(f"  Model file: {model_prefix}.model")
    print(f"  Vocab file: {model_prefix}.vocab")
    
    return f"{model_prefix}.model"

def load_tokenizer(model_path):

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"Vocabulary size: {sp.get_piece_size()}")
    print(f"Special tokens - PAD: {sp.pad_id()}, UNK: {sp.unk_id()}, BOS: {sp.bos_id()}, EOS: {sp.eos_id()}")
        
    return sp

def create_embedding_matrix(sp_model, embedding_dim=300):

    try:
        ft_en = api.load('fasttext-wiki-news-subwords-300')
        print("English FastText loaded")
    except:
        print("Failed to load English FastText")
        return None
        
    try:
        ft_fr = api.load('fasttext-wiki-news-subwords-300')  
        print("French FastText loaded")
    except:
        print("Failed to load French FastText")
        return None
    
    vocab = [sp_model.id_to_piece(i) for i in range(sp_model.get_piece_size())]
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize embedding matrix
    embedding_matrix = torch.randn(vocab_size, embedding_dim) * 0.1  # Small random init
    
    # Fill with FastText vectors
    found_en = found_fr = not_found = special_tokens = 0
    
    for i, piece in enumerate(vocab):
        # Handle special tokens
        if piece in ['<pad>', '<unk>', '<s>', '</s>', '<en>', '<fr>'] or piece.startswith('▁'):
            # Keep random initialization for special tokens and space markers
            special_tokens += 1
        elif piece in ft_en:
            embedding_matrix[i] = torch.tensor(ft_en[piece], dtype=torch.float32)
            found_en += 1
        elif piece in ft_fr:
            embedding_matrix[i] = torch.tensor(ft_fr[piece], dtype=torch.float32)
            found_fr += 1
        else:
            # Try without BPE markers
            clean_piece = piece.replace('▁', '').replace('##', '')
            if clean_piece in ft_en:
                embedding_matrix[i] = torch.tensor(ft_en[clean_piece], dtype=torch.float32)
                found_en += 1
            elif clean_piece in ft_fr:
                embedding_matrix[i] = torch.tensor(ft_fr[clean_piece], dtype=torch.float32)
                found_fr += 1
            else:
                not_found += 1
    
    print(f"\nEmbedding matrix statistics:")
    print(f"  Shape: {embedding_matrix.shape}")
    print(f"  Found in English FastText: {found_en} ({found_en/vocab_size*100:.1f}%)")
    print(f"  Found in French FastText: {found_fr} ({found_fr/vocab_size*100:.1f}%)")
    print(f"  Special tokens (random): {special_tokens} ({special_tokens/vocab_size*100:.1f}%)")
    print(f"  Not found (random): {not_found} ({not_found/vocab_size*100:.1f}%)")
    
    coverage = (found_en + found_fr) / vocab_size * 100
    print(f"  Total FastText coverage: {coverage:.1f}%")
    
    return embedding_matrix

def complete_setup(english_sentences, french_sentences):

    model_path = create_proper_sentencepiece_model(
        english_sentences, 
        french_sentences, 
        model_prefix='translation_bpe_v2'
    )
    
    sp = load_tokenizer(model_path)
    
    embedding_matrix = create_embedding_matrix(sp)
    

    
    print(f"  Tokenizer: {model_path}")
    print(f"  Vocab size: {sp.get_piece_size()}")
    print(f"  Embedding shape: {embedding_matrix.shape}")
    
    return sp, embedding_matrix



