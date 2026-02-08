from torch import nn
from transformers import AutoModelForSequenceClassification, DistilBertTokenizer
from functools import partial
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np


class LoraLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.scaling = alpha/rank

    def forward(self, tensor):
        return self.scaling* (tensor @ self.A @ self.B)



class LinearWithLora(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.alpha = alpha
        self.rank = rank
        self.lora = LoraLayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, tensor):
        return self.linear(tensor) + self.lora(tensor)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# LoRA configuration
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.5

# Apply LoRA to attention layers
assign_lora = partial(LinearWithLora, rank=lora_rank, alpha=lora_alpha)

for layer in model.distilbert.transformer.layer:
    layer.attention.q_lin = assign_lora(layer.attention.q_lin)
    layer.attention.k_lin = assign_lora(layer.attention.k_lin)
    layer.attention.v_lin = assign_lora(layer.attention.v_lin)

# Unfreeze LoRA parameters and classifier
for name, param in model.named_parameters():
    if ('lora' in name) or ('classifier' in name) or ("pre_classifier" in name):
        param.requires_grad = True


trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
print("trainable params:", sum(p.numel() for _,p in trainable))
print("trainable tensors:", [n for n,_ in trainable][:30])

model.to(device)

print(model)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Load SST-2 dataset
print("Loading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
train_dataset = dataset['train']
val_dataset = dataset['validation']


# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)


# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training setup
learning_rate = 2e-4
num_epochs = 3
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)


# Training function
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Validation function
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Training loop
print(f"\nStarting training for {num_epochs} epochs...")
best_val_acc = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    val_loss, val_acc = validate(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
        }, 'best_model_lora.pt')
        print(f"Best model saved with validation accuracy: {val_acc:.4f}")

        lora_state_dict = {
            name: param
            for name, param in model.named_parameters()
            if 'lora' in name.lower() and param.requires_grad
        }

        # Save adapter weights
        torch.save({
            'lora_state_dict': lora_state_dict,
            'lora_config': {
                'rank': lora_rank,
                'alpha': lora_alpha,
                'target_modules': ['q_lin','k_lin','v_lin' ]
            },
            'epoch': epoch,
            'val_accuracy': val_acc
        }, 'lora_adapter.pt')

        print(f"Saved {sum(p.numel() for p in lora_state_dict.values())} LoRA parameters")

print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
