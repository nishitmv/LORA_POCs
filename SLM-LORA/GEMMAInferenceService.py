import os

import pandas as pd
import torch
from datasets import Dataset
from flask import Flask, request, jsonify
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, Trainer, TrainingArguments

# Initialize Flask App
app = Flask(__name__)
local_model_path = "./models/gemma-3-1b-it"
huggingface_access_token = ""

print("Downloading model Gemma 3 1B")
model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

print(f"Model saved to {local_model_path}")

# --- Configuration ---
MODEL_ID = "./models/gemma-3-1b-it"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Warning: Tokenizer load failed: {e}")
    tokenizer = None

# --- Model Definition (Fixed Forward Pass) ---
from transformers import AutoModelForCausalLM


class GemmaMultiHeadClassifier(nn.Module):
    def __init__(self, model_id, num_type_labels, num_code_labels, lora_config=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_id,  token=huggingface_access_token)

        self.Gemma = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=None,
            token=huggingface_access_token
        )

        # print(f"VANILLA Gemma ARCHITECTURE : \n {self.Gemma}")

        if lora_config is not None:
            self.Gemma = get_peft_model(self.Gemma, lora_config)

            self.Gemma.print_trainable_parameters()

        # Two separate Head for Code and Type , hiodeen size of model , 1536 for Gemma .

        self.type_head = nn.Linear(self.config.hidden_size, num_type_labels)
        self.code_head = nn.Linear(self.config.hidden_size, num_code_labels)

        # changing heads added to Gemma Data Type DFloat16
        self.type_head.to(self.Gemma.dtype)
        self.code_head.to(self.Gemma.dtype)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None,
                labels_type=None, labels_code=None, labels=None, **kwargs):

        if labels is not None:
            labels_type = labels[:, 0]  # Assuming first col is type
            labels_code = labels[:, 1]  # Assuming second col is code

        outputs = self.Gemma(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        # Extract Last Token Embedding (EOS token), for LLMs, use Last Hidden State of last token
        # shape : [batch, seq_length, hidden]
        # AutoModelForCausalLM might not return 'last_hidden_state' attribute directly in some versions,
        # but 'hidden_states' tuple is always there if output_hidden_states=True.

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            # Fallback: Get the last layer from the hidden_states tuple
            last_hidden_state = outputs.hidden_states[-1]

        #print( "Shape of last hidden state %",last_hidden_state.shape )
        #Shape of last hidden state = torch.Size([32, 64, 1536])
        # Get Embedding of last token for Classification
        pooled_output = last_hidden_state[:, -1, :]


        # Get the Vector for last token in the sequence using sequence lenght calced above
        #last_hidden_state shape: (Batch_Size, Sequence_Length, Hidden_Size)
        # last_hidden_state[0] = batch size
        # sequence_lengths contains last token indexes for each sequence .
        # Last token is sequence is like CLS token it has learnt about the sequence
        #pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), sequence_lengths]

        # Pass it to the linear layers like we do CLS Token in Transforemrs
        logits_type = self.type_head(pooled_output)
        logits_code = self.code_head(pooled_output)

        loss = None

        if labels_type is not None and labels_code is not None:
            loss_type = self.loss_fn(logits_type, labels_type)
            loss_code = self.loss_fn(logits_code, labels_code)
            loss = 2 * loss_type + 1 * loss_code
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        # Form output that can work with Huggingface trainer
        output = {
            "logits": (logits_type, logits_code)
        }

        return loss, logits_type, logits_code




# --- Internal Helper: Data Loading ---
def _get_company_dataset(data_file_path, tokenizer, max_length=64):
    """
    Internal function to read company data, ignore extra fields, and tokenizing.
    """
    # 1. Read Data
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"File not found: {data_file_path}")

    df = pd.read_csv(data_file_path)

    # 2. Strict Column Filtering
    required_cols = ['merchant_group', 'merchant_name','merchant_category', 'amount', 'currency']
    # Filter to keep ONLY the required columns, ignoring everything else
    # We use intersection to be safe, or direct selection if we enforce existence
    df = df[required_cols]

    # 3. Tokenization Logic
    dataset = Dataset.from_pandas(df, preserve_index=False)

    def bucketize_amount(amount):
        try:
            val = float(amount)
            # Handle negative (refunds/credits) separately if needed
            is_negative = val < 0
            val = abs(val)

            if val == 0:
                bucket = "ZERO"
            elif val < 10:
                bucket = "XS"  # e.g., < $10 (Coffee, fast food)
            elif val < 50:
                bucket = "SMALL"  # e.g., $10-$50 (Groceries, gas)
            elif val < 100:
                bucket = "MEDIUM"  # e.g., $50-$100 (Utilities, clothes)
            elif val < 500:
                bucket = "LARGE"  # e.g., $100-$500 (Electronics, furniture)
            elif val < 1000:
                bucket = "XL"  # e.g., $500-$1000 (Rent, high-end)
            else:
                bucket = "XXL"  # e.g., > $1000 (B2B payments, salary)

            prefix = "REFUND_" if is_negative else ""
            return f"{prefix}{bucket}"
        except (ValueError, TypeError):
            return "UNKNOWN"

    def preprocess_function(examples):
        # Manual concatenation for GEMMA/LLMs
        inputs = [
            f"Merchant: {name} | Group: {group} | Category: {cat} | Amount: {bucketize_amount(amt)} | currency: {curr} "
            for name, group, cat, amt, curr in zip(
                examples["merchant_name"],
                examples["merchant_group"],
                examples["merchant_category"],
                examples["amount"],
                examples["currency"],
            )
        ]

        tokenized_inputs = tokenizer(
            inputs,  # Single list of strings
            truncation=True,
            max_length=max_length,
            padding="max_length"  # Or False if using DataCollator
        )

        #tokenized_inputs["labels_type"] = examples["cc_type_id"]
        #tokenized_inputs["labels_code"] = examples["cc_code_id"]
        return tokenized_inputs



    # Remove text columns to leave only tensors
    remove_cols = dataset.column_names
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)
    target_columns = ["input_ids", "attention_mask"]
    # Set format for PyTorch
    # We only need input_ids and attention_mask for inference
    dataset.set_format(type="torch", columns=target_columns)

    return dataset


# --- Helper: Model Loading (Placeholder) ---
def load_model_for_company(company, dataset):

    print(f"Loading model for {company}...")
    peft_config = LoraConfig(
        #  treat the GEMMA backbone as a feature extractor.as we have custom multihead GEMMA ,
        # normal classification this would have been SEQ_CLS
        # classification heads are external to the PEFT wrapper here
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,  # 16 RANk is good , LORA Mattrices will be A X R and R X B .
        lora_alpha=32,
        # Scales Output of Lora adapter by Alpha / Rank . ( 32/16 for us) , Makes learnt weights LOUDER compared to base model weights .
        # Scale of 2 is good .
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # Q K V and Output Projections selected to train as part of adapter
        # MLP Layers are gate_proj, up_proj, down_proj , but results in very large number of paramters to learn but also gives huge accuracy benefit .
        lora_dropout=0.05  # Prevents overfittig whern data sizes are small like our company data case.
    )

    model = GemmaMultiHeadClassifier(
        model_id=MODEL_ID,
        num_type_labels=2,  # Dummy
        num_code_labels=2  # Dummy
    )
    model.Gemma.resize_token_embeddings(len(tokenizer))

    # CRITICAL FIX 2: Wrap in PEFT immediately
    # This adds the "base_model.model..." structure so load_adapter works
    model.Gemma = get_peft_model(model.Gemma, peft_config)

    # Move to GPU once
    model.to("cuda")
    print(f"GEMMA Multihead with 2 linear heads :\n {model}")
    adapter_path = f"./final_adapters_Gemma/{company}"
    adapter_name = f"adapter_{company}"
    try:
        # Load state dicts to cpu to inspect shapes
        type_state = torch.load(os.path.join(adapter_path, "type_head.bin"), map_location="cpu")
        code_state = torch.load(os.path.join(adapter_path, "code_head.bin"), map_location="cpu")

        n_types = type_state['weight'].shape[0]
        n_codes = code_state['weight'].shape[0]

        # Resize the linear layers on the model
        # We reuse the existing dtype/device
        dtype = model.Gemma.dtype
        device = model.Gemma.device

        model.type_head = torch.nn.Linear(model.config.hidden_size, n_types).to(device=device, dtype=dtype)
        model.code_head = torch.nn.Linear(model.config.hidden_size, n_codes).to(device=device, dtype=dtype)

        # Load weights
        model.type_head.load_state_dict(type_state)
        model.code_head.load_state_dict(code_state)

    except FileNotFoundError:
        print(f"Skipping Global Adapter: Head weights not found.")
        return None

    # B. Load LoRA Adapter
    try:
        # load_adapter adds the weights.
        # Since we already wrapped in get_peft_model, this should match keys correctly.
        model.Gemma.load_adapter(adapter_path, adapter_name)
        model.Gemma.set_adapter(adapter_name)
        print("MODEL READY")
        return model
    except Exception as ex:
        print(f"Skipping Global Adapter Loading : Adapter load failed. {ex}")
        return None

def preprocess_logits_for_metrics(logits, labels):
    # Ensure this returns a tuple/tensor, NOT None!
    if isinstance(logits, tuple):
        return (logits[1], logits[2]) # (logits_type, logits_code)
    # Fallback debug
    print(f"DEBUG: preprocess received {type(logits)}")
    return logits


import numpy as np  # Make sure this is imported at the top


@app.route('/predict', methods=['POST'])
def predict():
    """
    Input JSON: { "company": "CompanyA", "data_file_path": "./data/test.csv" }
    Output JSON: { "company": "CompanyA", "predictions": [ {"merchant_group": "...", "merchant_name": "...", "cc_type_id": 1, "cc_code_id": 5}, ... ] }
    """
    #Val_Data_company_a.csv
    try:
        data = request.get_json()
        company = data.get('company')
        data_file_path = data.get('data_file_path')

        if not company or not data_file_path:
            return jsonify({"error": "Missing parameters"}), 400

        if not os.path.exists(data_file_path):
            return jsonify({"error": f"File not found: {data_file_path}"}), 404

        # 1. Read Original Data to keep text labels (aligns row-by-row with dataset)
        original_df = pd.read_csv(data_file_path)

        # Basic validation to ensure columns exist
        if 'merchant_group' not in original_df.columns or 'merchant_name' not in original_df.columns:
            return jsonify({"error": "CSV missing 'merchant_group' or 'merchant_name' columns"}), 400

        # 2. Get Dataset (Tensor format for model)
        # Note: This reads the file again, but ensures correct tokenization
        dataset = _get_company_dataset(data_file_path, tokenizer)

        # 3. Load Model
        # WARNING: Loading a fresh model on every request is slow.
        # Ideally, load model once globally and just switch adapters here.
        model = load_model_for_company(company, dataset)
        if model is None:
            return jsonify({"error": f"Failed to load model for {company}"}), 500

        # 4. Run Inference
        eval_trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./temp_eval",
                per_device_eval_batch_size=32,
                remove_unused_columns=False,
                report_to="none"
            ),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Get raw logits
        preds = eval_trainer.predict(dataset, ignore_keys=["loss", "hidden_states", "attentions"])

        # 5. Extract Predictions
        # preds.predictions is a tuple (logits_type, logits_code)
        logits_type, logits_code = preds.predictions

        # Convert Logits -> Class IDs (Integers)
        pred_type_ids = np.argmax(logits_type, axis=1)
        pred_code_ids = np.argmax(logits_code, axis=1)

        # 6. Zip Original Text with Predictions
        results = []
        for idx in range(len(dataset)):
            results.append({
                "merchant_group": str(original_df.iloc[idx]['merchant_group']),
                "merchant_name": str(original_df.iloc[idx]['merchant_name']),
                "merchant_category": str(original_df.iloc[idx]['merchant_category']),
                "amount": str(original_df.iloc[idx]['amount']),
                "currency": str(original_df.iloc[idx]['currency']),
                "cc_type_id": int(pred_type_ids[idx]),
                "cc_code_id": int(pred_code_ids[idx])
            })

        return jsonify({
            "company": company,
            "predictions": results
        })

    except Exception as ex:
        # Print stack trace to console for debugging
        print(f"Prediction Error: {ex}")
        return jsonify({"error": str(ex)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
#Val_Data_company_a.csv