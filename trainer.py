import torch
import bleu_tester as bt

def train_step_fixed(model, src_batch, tgt_batch, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    tgt_input = tgt_batch[:, :-1]  # Remove last token (usually EOS) for input
    tgt_output = tgt_batch[:, 1:]  # Remove first token (BOS) for target
    
    logits = model(src_batch, tgt_input)  # [batch_size, tgt_seq_len-1, vocab_size]
    
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
        tgt_output.reshape(-1)                # [batch_size * seq_len]
    )
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def test_step(model, src_batch, tgt_batch, criterion):
    model.eval()
    with torch.no_grad():
        tgt_input = tgt_batch[:, :-1]  # Remove last token for input
        tgt_output = tgt_batch[:, 1:]  # Remove first token for target
        
        logits = model(src_batch, tgt_input)  # [batch_size, tgt_seq_len-1, vocab_size]
        
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
            tgt_output.reshape(-1)                # [batch_size * seq_len]
        )
    
    return loss.item()

def train(model, train_loader, test_loader, num_epochs, device, optimizer, criterion, sp):
    sources, references = bt.get_sentences()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            loss = train_step_fixed(model, src_batch, tgt_batch, optimizer, criterion)
            total_train_loss += loss
            
            if batch_idx % 1000 == 0:
                print(f"Batch {batch_idx}, Training Loss: {loss:.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        
        # Testing phase
        model.eval()
        total_test_loss = 0
        for src_batch, tgt_batch in test_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            loss = test_step(model, src_batch, tgt_batch, criterion)
            total_test_loss += loss
        
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Average Test Loss: {avg_test_loss:.4f}")
        
        # Evaluate BLEU score on test set (assuming test sources/references are same for simplicity)
        test_results = bt.calculate_bleu_score(model, sp, sources, references)
        print(f"Test BLEU Score: {test_results['bleu_score']:.4f}")
    
    return model