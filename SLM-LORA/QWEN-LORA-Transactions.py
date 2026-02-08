import os

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoConfig
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments


def get_company_dataset(file_path, tokenizer, max_length=64, test_size=0.2):
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
        # Manual concatenation for Qwen/LLMs
        inputs = [f"{g} | {n}" for g, n in zip(examples["merchant_group"], examples["merchant_name"])]

        tokenized_inputs = tokenizer(
            inputs,  # Single list of strings
            truncation=True,
            max_length=max_length,
            padding="max_length"  # Or False if using DataCollator
        )

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

class QwenMultiHeadClassifier(nn.Module):
    def __init__(self, model_id, num_type_labels, num_code_labels, lora_config=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_id)
        self.qwen = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map='auto'
        )

        if lora_config is not None:
            self.qwen = get_peft_model(self.qwen, lora_config)
            self.qwen.print_trainable_parameters()

        # Two separate Head for Code and Type , hiodeen size of model , 1536 for Qwen .

        self.type_head = nn.Linear(self.config.hidden_size, num_type_labels)
        self.code_head = nn.Linear(self.config.hidden_size, num_code_labels)

        self.type_head.to(self.qwen.dtype)
        self.code_head.to(self.qwen.dtype)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None,
                    labels_type = None, labels_code = None, **kwargs):
            outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
            # Extract Last Token Embedding (EOS token), for LLMs, use Last Hidden State of last token
            # shape : [batch, seq_length, hidden]
            last_hidden_state = outputs.last_hidden_state
            #print( "Shape of last hidden state %",last_hidden_state.shape )
            #Shape of last hidden state = torch.Size([32, 64, 1536])
            # Get Embedding of las token for Classification
            if self.config.pad_token_id is None:  # Fallback if no pad token
                sequence_lengths = -1
            else:
                if input_ids is not None:  # Find last non paddign token
                    sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(last_hidden_state.device)
                else:
                    sequence_lengths = -1

            # Get the Vector for last token in the sequence using sequence lenght calced above
            #last_hidden_state shape: (Batch_Size, Sequence_Length, Hidden_Size)
            # last_hidden_state[0] = batch size
            # sequence_lengths conatins last token indexes for each sequence .
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), sequence_lengths]

            logits_type = self.type_head(pooled_output)
            logits_code = self.code_head(pooled_output)

            loss = None

            if labels_type is not None and labels_code is not None:
                loss_type = self.loss_fn(logits_type, labels_type)
                loss_code = self.loss_fn(logits_code, labels_code)
                loss = 2 * loss_type + 1 * loss_code

            # Form output that can work with Huggingface trainer

            return {
                "loss": loss,
                "logits": (logits_type, logits_code),
                # Trainer accepts tuples here if you handle them in compute_metrics
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }


class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        # 1. Save the LoRA adapters (standard behavior)
        # Checks if model is wrapped in PEFT
        if output_dir is None:
            output_dir = self.args.output_dir

        # Save LoRA weights
        self.model.qwen.save_pretrained(output_dir)

        # 2. MANUALLY save your custom heads
        torch.save(self.model.type_head.state_dict(), f"{output_dir}/type_head.bin")
        torch.save(self.model.code_head.state_dict(), f"{output_dir}/code_head.bin")

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)


# 1. Setup
model_id = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id,  trust_remote_code=True,  use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


companies = ["company_D", "company_E", "company_F"]  # Your list of companies


# 2. Initialize Base Model Wrapper
# We initialize it ONCE. The base weights remain frozen; only adapters change.

# 3. Define LoRA Config
# modules_to_save is CRITICAL here. It ensures your custom heads are trainable.
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # Using custom model, so generic task
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05
)


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

    model = QwenMultiHeadClassifier(
        model_id=model_id,
        num_type_labels=num_type_labels,
        num_code_labels=num_code_labels,
        lora_config=peft_config
    )

    # B. Manage Adapters
    # If it's the first run, the 'default' adapter is active.
    # For subsequent runs, we add a new adapter.
    adapter_name = f"adapter_{company}"

    try:
        model.qwen.add_adapter(adapter_name=adapter_name, peft_config=peft_config)
    except ValueError:
        pass  # Adapter might already exist if resuming

    # C. Train
    training_args = TrainingArguments(
        output_dir=f"./results/{company}",
        per_device_train_batch_size=32,
        num_train_epochs=3,
        save_strategy="no",  # We save manually to be safe
        learning_rate=2e-4,
        remove_unused_columns=False  # Important for custom models
    )

    custom_trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    custom_trainer.train()

    metrics = custom_trainer.evaluate(eval_dataset=val_dataset)
    print(metrics)


    save_path = f"./final_adapters/{company}"
    # D. Save Adapter & Heads
    # This saves the LoRA weights AND the 'modules_to_save' (heads) to the folder
    model.qwen.save_pretrained(save_path)
    torch.save(model.type_head.state_dict(), os.path.join(save_path, "type_head.bin"))
    torch.save(model.code_head.state_dict(), os.path.join(save_path, "code_head.bin"))
    # E. Cleanup to free VRAM for next company
    # delete_adapter removes the LoRA weights from memory
    # Note: 'modules_to_save' weights might persist in the base model state dict
    # if not carefully handled, but delete_adapter handles the PEFT part.
    model.qwen.delete_adapter(adapter_name)


