from transformers import EncoderDecoderModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import evaluate
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
from collections import Counter

# ============================
# CONFIGURATION
# ============================
class Config:
    BATCH_SIZE = 32
    EPOCHS = 35
    LEARNING_RATE = 9e-5
    MAX_LENGTH = 100
    WARMUP_STEPS = 2000
    GRADIENT_ACCUMULATION_STEPS = 2
    PATIENCE = 15  # Early stopping patience
    FREEZE_ENCODER = True
    FREEZE_DECODER = True
    TRAIN_CROSS_ATTENTION = True
    
config = Config()
# ============================
# TOKENIZER AND MODEL SETUP
# ============================
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

print("Loading model...")
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'bert-base-multilingual-cased',
    'bert-base-multilingual-cased'
)

# Configure generation parameters
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# ============================
# IMPROVED PARAMETER FREEZING
# ============================
def setup_parameter_freezing(model, config):
    """Properly freeze/unfreeze model parameters"""
    trainable_params = []
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze based on configuration
    if not config.FREEZE_ENCODER:
        for param in model.encoder.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("Encoder parameters are trainable")
    
    if not config.FREEZE_DECODER:
        for param in model.decoder.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("Decoder parameters are trainable")
    
    # Unfreeze cross-attention layers
    if config.TRAIN_CROSS_ATTENTION:
        # Look for cross-attention parameters
        for name, param in model.named_parameters():
            if any(keyword in name.lower() for keyword in ['cross', 'attention']):
                param.requires_grad = True
                trainable_params.append(param)
        
        print(f"Training {len(trainable_params)} cross-attention related parameters")
    
    return trainable_params

trainable_params = setup_parameter_freezing(model, config)

# ============================
# MEMORY-EFFICIENT DATASET
# ============================
class EfficientTranslationDataset(Dataset):
    def __init__(self, source_path, target_path, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Store file paths instead of loading all data
        self.source_path = source_path
        self.target_path = target_path
        
        # Count lines efficiently
        self.num_samples = self._count_lines()
        print(f"Found {self.num_samples} translation pairs")
    
    def _count_lines(self):
        """Count lines in file without loading all content"""
        count = 0
        with open(self.source_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Read only the needed line
        with open(self.source_path, 'r', encoding='utf-8') as f_src:
            for i, line in enumerate(f_src):
                if i == idx:
                    source_text = line.strip()
                    break
        
        with open(self.target_path, 'r', encoding='utf-8') as f_tgt:
            for i, line in enumerate(f_tgt):
                if i == idx:
                    target_text = line.strip()
                    break
        
        # Tokenize
        source_encodings = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encodings['input_ids'].squeeze(),
            'attention_mask': source_encodings['attention_mask'].squeeze(),
            'labels': target_encodings['input_ids'].squeeze()
        }

# ============================
# IMPROVED TRAINING FUNCTION
# ============================
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, config, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Replace pad_token_id with -100 for loss calculation
        labels[labels == tokenizer.pad_token_id] = -100
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'lr': scheduler.get_last_lr()[0]
        })
    
    return total_loss / len(dataloader)

# ============================
# GEOMETRIC MEAN BLEU CALCULATION
# ============================
def calculate_geometric_bleu(predictions, references):
    """Calculate BLEU score using geometric mean of n-gram precisions"""
    if not predictions or not references:
        return {'bleu': 0}
    
    # Tokenize
    tokenized_preds = []
    tokenized_refs = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if pred_tokens or ref_tokens:
            tokenized_preds.append(pred_tokens)
            tokenized_refs.append([ref_tokens])  # Wrap in list for multiple references support
    
    if not tokenized_preds:
        return {'bleu': 0}
    
    # Import nltk BLEU scorer
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        chencherry = SmoothingFunction()
        
        # Calculate BLEU score with geometric mean
        bleu_score = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=(0.25, 0.25, 0.25, 0.25),  # Equal weights for 1-4 grams
            smoothing_function=chencherry.method1
        )
        
    except ImportError:
        print("NLTK not installed, falling back to simplified BLEU calculation")
        # Fallback to simplified calculation
        bleu_score = calculate_simple_geometric_bleu(tokenized_preds, tokenized_refs)
    
    return {'bleu': bleu_score}

def calculate_simple_geometric_bleu(predictions, references_list):
    """Simplified geometric BLEU calculation"""
    max_n = 4
    precisions = []
    
    # Calculate precision for each n-gram
    for n in range(1, max_n + 1):
        total_matches = 0
        total_pred_ngrams = 0
        
        for pred_tokens, ref_list in zip(predictions, references_list):
            if len(pred_tokens) < n:
                continue
            
            # Generate n-grams for prediction
            pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
            if not pred_ngrams:
                continue
            
            # Generate n-grams for all references
            ref_ngrams_list = []
            for ref_tokens in ref_list:
                ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
                ref_ngrams_list.append(ref_ngrams)
            
            # Count matches using closest reference length (modified precision)
            for ngram in pred_ngrams:
                # Count in prediction
                pred_count = pred_ngrams.count(ngram)
                # Maximum count in any reference
                max_ref_count = max([ref_ngrams.count(ngram) for ref_ngrams in ref_ngrams_list])
                
                total_matches += min(pred_count, max_ref_count)
            total_pred_ngrams += len(pred_ngrams)
        
        if total_pred_ngrams > 0:
            precisions.append(total_matches / total_pred_ngrams)
        else:
            precisions.append(0)
    
    # Brevity penalty
    c = sum(len(p) for p in predictions)
    r = min(sum(len(r[0]) for r in references_list), c)  # Use closest reference length
    
    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - r/c) if c > 0 else 0
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bleu_score = bp * geometric_mean
    
    return bleu_score

# ============================
# VALIDATION FUNCTION
# ============================
def validate(model, dataloader, device, tokenizer, config, split_name="Validation"):
    """Validation with optimized generation"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=split_name):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Calculate loss
            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == tokenizer.pad_token_id] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_for_loss,
                return_dict=True
            )
            total_loss += outputs.loss.item()
            
            # Generate predictions with optimized settings
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.MAX_LENGTH,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                no_repeat_ngram_size=3,
                decoder_start_token_id=model.config.decoder_start_token_id,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id
            )
            
            # Decode batch efficiently
            pred_texts = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            ref_texts = tokenizer.batch_decode(
                labels, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            all_predictions.extend([t.strip() for t in pred_texts])
            all_references.extend([t.strip() for t in ref_texts])
    
    # Calculate BLEU scores using geometric mean
    bleu_scores = calculate_geometric_bleu(all_predictions, all_references)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, bleu_scores, all_predictions, all_references

# ============================
# SETUP DATA AND MODEL (WITH TEST SPLIT)
# ============================
print("\nCreating dataset...")
dataset = EfficientTranslationDataset(
    source_path="source.txt",
    target_path="target.txt",
    tokenizer=tokenizer,
    max_length=config.MAX_LENGTH
)

# Split dataset into train/validation/test (60/20/20)
total_size = len(dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

# Create splits
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

# ============================
# TRAINING SETUP
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"\nUsing device: {device}")

# Optimizer and scheduler
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(
    trainable_parameters,
    lr=config.LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Learning rate scheduler
total_steps = len(train_dataloader) * config.EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
warmup_steps = min(config.WARMUP_STEPS, total_steps // 10)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ============================
# TRAINING LOOP WITH EARLY STOPPING
# ============================
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

best_bleu = 0
patience_counter = 0
training_history = []

# Store losses for plotting
train_losses = []
val_losses = []
val_bleus = []

for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
    print("-" * 50)
    
    # Training
    train_loss = train_epoch(
        model, train_dataloader, optimizer, scheduler, 
        device, epoch, config, config.GRADIENT_ACCUMULATION_STEPS
    )
    train_losses.append(train_loss)
    
    # Validation
    val_loss, bleu_scores, _, _ = validate(
        model, val_dataloader, device, tokenizer, config, "Validation"
    )
    val_losses.append(val_loss)
    val_bleus.append(bleu_scores['bleu'])
    
    # Log results
    epoch_stats = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_bleu': bleu_scores['bleu']
    }
    training_history.append(epoch_stats)
    
    # Print results
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val BLEU: {bleu_scores['bleu']:.4f}")
    
    # Save best model
    if bleu_scores['bleu'] > best_bleu:
        best_bleu = bleu_scores['bleu']
        os.makedirs("best_model", exist_ok=True)
        model.save_pretrained("best_model")
        tokenizer.save_pretrained("best_model")
        
        # Also save training history
        torch.save(training_history, "best_model/training_history.pt")
        
        print(f"  ✓ New best model! BLEU: {bleu_scores['bleu']:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{config.PATIENCE}")
    
    # Early stopping
    if patience_counter >= config.PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

# ============================
# FINAL TEST EVALUATION
# ============================
print("\n" + "="*50)
print("FINAL TEST EVALUATION")
print("="*50)

# Load best model for testing
print("Loading best model for testing...")
model = EncoderDecoderModel.from_pretrained("best_model")
model = model.to(device)

# Evaluate on test set
test_loss, test_bleu_scores, test_predictions, test_references = validate(
    model, test_dataloader, device, tokenizer, config, "Test"
)

print(f"\nTest Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test BLEU: {test_bleu_scores['bleu']:.4f}")

# Save test results
test_results = {
    'test_loss': test_loss,
    'test_bleu': test_bleu_scores['bleu'],
    'predictions': test_predictions[:10],  # Save first 10 examples
    'references': test_references[:10]
}

torch.save(test_results, "best_model/test_results.pt")

# Print some examples
print("\nSample predictions:")
for i in range(min(5, len(test_predictions))):
    print(f"  Reference: {test_references[i]}")
    print(f"  Prediction: {test_predictions[i]}")
    print()

# ============================
# PLOTTING FUNCTIONS
# ============================
def plot_training_history(train_losses, val_losses, val_bleus):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot BLEU scores
    axes[1].plot(epochs, val_bleus, 'g-', label='Validation BLEU', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BLEU Score')
    axes[1].set_title('Validation BLEU Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_final_summary(training_history, test_results):
    """Create a summary plot of all metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Extract data
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    val_bleus = [h['val_bleu'] for h in training_history]
    
    # Plot 1: Loss curves
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: BLEU score progression
    axes[1].plot(epochs, val_bleus, 'g-', linewidth=2)
    axes[1].scatter(epochs, val_bleus, c='g', s=20)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BLEU Score')
    axes[1].set_title('Validation BLEU Score Progression')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Final scores comparison
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_val_bleu = val_bleus[-1]
    
    # Bar chart for losses
    x_pos = [0, 1, 2]
    metrics = ['Train Loss', 'Val Loss', 'Test Loss']
    values = [final_train_loss, final_val_loss, test_results['test_loss']]
    
    bars = axes[2].bar(x_pos, values, color=['blue', 'red', 'orange'], alpha=0.7)
    axes[2].set_xlabel('Dataset')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Final Loss Comparison')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(metrics)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Add BLEU score as text
    axes[2].text(1.5, max(values) * 0.9, 
                f'Test BLEU: {test_results["test_bleu"]:.4f}',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('final_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================
# GENERATE AND SAVE PLOTS
# ============================
print("\n" + "="*50)
print("GENERATING PLOTS")
print("="*50)

# Create plots
plot_training_history(train_losses, val_losses, val_bleus)
plot_final_summary(training_history, test_results)

# ============================
# FINAL SUMMARY
# ============================
print("\n" + "="*50)
print("TRAINING COMPLETED - FINAL SUMMARY")
print("="*50)
print(f"Total epochs trained: {len(training_history)}")
print(f"\nBest Validation BLEU: {best_bleu:.4f}")
print(f"Final Test BLEU: {test_results['test_bleu']:.4f}")
print(f"Final Test Loss: {test_results['test_loss']:.4f}")

# Save final model
os.makedirs("final_model", exist_ok=True)
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")

# Save complete training history
final_history = {
    'config': config.__dict__,
    'training_history': training_history,
    'test_results': test_results,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_bleus': val_bleus
}

torch.save(final_history, "final_model/complete_history.pt")
print("\nFinal model and complete history saved to 'final_model/'")
print("Plots saved as 'training_history.png' and 'final_summary.png'")
