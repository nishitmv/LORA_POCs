import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import  Trainer, TrainingArguments


def get_company_dataset(file_path, tokenizer, max_length=128, test_size=0.2):
    """
    Reads a CSV file, processes it for ModernBERT Multi-Head classification,
    and returns Train/Val datasets + dimensions for the model heads.
    """
    # 1. Load Data
    df = pd.read_csv(f"{company}.csv")

    # 2. Label Encoding (Local to this company/dataset)
    # If you need global consistency across companies, pass fitted encoders instead.
    type_encoder = LabelEncoder()
    code_encoder = LabelEncoder()

    df['cc_type_id'] = type_encoder.fit_transform(df['cc_type'])
    df['cc_code_id'] = code_encoder.fit_transform(df['cc_code'])

    # Calculate dimensions for the model heads
    num_type_labels = len(type_encoder.classes_)
    num_code_labels = len(code_encoder.classes_)

    print(f"Dataset Loaded: {len(df)} records")
    print(f"Found {num_type_labels} Transaction Types and {num_code_labels} GL Codes.")

    # 3. Stratified Train/Test Split
    # We create a temporary 'stratify_col' to ensure both Type and Code distributions are preserved
    df['stratify_col'] = df['cc_type'].astype(str) + "_" + df['cc_code'].astype(str)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df['stratify_col']  # Critical for rare GL codes
    )

    # Cleanup auxiliary columns
    cols_to_keep = ['merchant_group', 'merchant_name', 'cc_type_id', 'cc_code_id']
    train_df = train_df[cols_to_keep]
    val_df = val_df[cols_to_keep]

    # 4. Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

    # 5. Tokenization Function
    def preprocess_function(examples):
        # Create input: "[CLS] Merchant Group [SEP] Merchant Name [SEP]"
        tokenized_inputs = tokenizer(
            text=examples["merchant_group"],
            text_pair=examples["merchant_name"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        # Map to specific arguments expected by our custom ModernBertMultiHead
        tokenized_inputs["labels_type"] = examples["cc_type_id"]
        tokenized_inputs["labels_code"] = examples["cc_code_id"]

        return tokenized_inputs

    # 6. Apply Processing
    # We remove the text columns to leave only the tensors
    remove_cols = train_dataset.column_names

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)

    # 7. Set Format for PyTorch
    target_columns = ["input_ids", "attention_mask", "labels_type", "labels_code"]
    train_dataset.set_format(type="torch", columns=target_columns)
    val_dataset.set_format(type="torch", columns=target_columns)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "num_type_labels": num_type_labels,
        "num_code_labels": num_code_labels,
        "encoders": {"type": type_encoder, "code": code_encoder}
    }


class ModernBertMultiHead(nn.Module):
    def __init__(self, model_name, num_type_labels, num_code_labels):
        super().__init__()
        # Load base ModernBERT model
        self.bert = AutoModel.from_pretrained(model_name)

        self.config = self.bert.config

        hidden_size = self.bert.config.hidden_size
        print(hidden_size)
        # Define two separate heads
        self.type_head = nn.Linear(hidden_size, num_type_labels)
        self.code_head = nn.Linear(hidden_size, num_code_labels)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels_type=None, labels_code=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # ModernBERT typically uses Mean Pooling or [CLS] (index 0)
        # We use index 0 (CLS-equivalent) for classification
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Get logits from both heads
        logits_type = self.type_head(pooled_output)
        logits_code = self.code_head(pooled_output)

        loss = None
        if labels_type is not None and labels_code is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_type = loss_fct(logits_type, labels_type)
            loss_code = loss_fct(logits_code, labels_code)
            loss = loss_type + loss_code  # Sum losses

        return {"loss": loss, "logits_type": logits_type, "logits_code": logits_code}




# 1. Setup
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
companies = ["company_A", "company_B", "company_C"]  # Your list of companies

# Define Label Counts (assuming these are global, otherwise calculate per loop)
NUM_CC_TYPES = 5  # e.g., Visa, MC, Amex
NUM_CC_CODES = 100  # e.g., MCC codes


# 2. Initialize Base Model Wrapper
# We initialize it ONCE. The base weights remain frozen; only adapters change.

# 3. Define LoRA Config
# modules_to_save is CRITICAL here. It ensures your custom heads are trainable.
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # Using custom model, so generic task
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["Wqkv", "Wo", "Wi", "W2"],  # ModernBERT target modules (verify exact names)
    modules_to_save=["type_head", "code_head"]  # TRAIN THESE LAYERS
)

# Apply PEFT wrapper

# 4. Training Loop
for company in companies:
    print(f"Training adapter for: {company}")

    # A. Get Data for this Company
    # Assume get_company_dataset returns a formatted HF Dataset
    # Format: "Merchant Group: {grp} [SEP] Merchant Name: {name}"
    # 1. Get the data bundle
    data_bundle = get_company_dataset(company, tokenizer)


    # 2. Extract components
    train_dataset = data_bundle["train"]
    val_dataset = data_bundle["val"]
    num_type_labels = data_bundle["num_type_labels"]
    num_code_labels = data_bundle["num_code_labels"]
    base_model = ModernBertMultiHead(model_id, num_type_labels, num_code_labels)
    model = get_peft_model(base_model, peft_config)
    # B. Manage Adapters
    # If it's the first run, the 'default' adapter is active.
    # For subsequent runs, we add a new adapter.
    adapter_name = f"adapter_{company}"

    try:
        model.add_adapter(adapter_name, peft_config)
    except ValueError:
        pass  # Adapter might already exist if resuming

    model.set_adapter(adapter_name)

    # C. Train
    training_args = TrainingArguments(
        output_dir=f"./results/{company}",
        per_device_train_batch_size=32,
        num_train_epochs=3,
        save_strategy="no",  # We save manually to be safe
        learning_rate=2e-4,
        remove_unused_columns=False  # Important for custom models
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    metrics = trainer.evaluate(eval_dataset=val_dataset)
    print(metrics)
    # Quick Manual Check
    model.eval()
    with torch.no_grad():
        # Grab a single batch from validation
        batch = next(iter(trainer.get_eval_dataloader()))
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)

        # Get predictions
        pred_type = torch.argmax(outputs['logits_type'], dim=1)
        pred_code = torch.argmax(outputs['logits_code'], dim=1)

        print("True Types:", batch['labels_type'][:5])
        print("Pred Types:", pred_type[:5])
        print("True Codes:", batch['labels_code'][:5])
        print("Pred Codes:", pred_code[:5])


    # D. Save Adapter & Heads
    # This saves the LoRA weights AND the 'modules_to_save' (heads) to the folder
    model.save_pretrained(f"./final_adapters/{company}")

    # E. Cleanup to free VRAM for next company
    # delete_adapter removes the LoRA weights from memory
    # Note: 'modules_to_save' weights might persist in the base model state dict
    # if not carefully handled, but delete_adapter handles the PEFT part.
    model.delete_adapter(adapter_name)


