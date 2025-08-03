import torch
import bleu_tester as bt


def train_step_fixed(model, src_batch, tgt_batch, optimizer, criterion):
    """
    Training step that properly handles BOS/EOS tokens
    """
    model.train()
    optimizer.zero_grad()
    
    # Prepare target sequences for teacher forcing
    # Input: full target with BOS, Output: target without BOS
    tgt_input = tgt_batch[:, :-1]  # Remove last token (usually EOS) for input
    tgt_output = tgt_batch[:, 1:]  # Remove first token (BOS) for target
    
    # Forward pass
    logits = model(src_batch, tgt_input)  # [batch_size, tgt_seq_len-1, vocab_size]
    
    # Reshape for loss calculation
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
        tgt_output.reshape(-1)                # [batch_size * seq_len]
    )
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


def train(model , dataloader , num_epochs , device , optimizer , criterion , sp):
    sources, references = bt.get_sentences()
    for epoch in range(num_epochs):
        
        results = bt.calculate_bleu_score(model, sp, sources, references)
        print(f"BLEU Score: {results['bleu_score']:.4f}")
        
        total_loss = 0
        for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            loss = train_step_fixed(model, src_batch, tgt_batch, optimizer, criterion)
            total_loss += loss
            
            if batch_idx % 1000 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')