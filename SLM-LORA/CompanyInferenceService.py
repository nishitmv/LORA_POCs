import os
import pandas as pd
import torch
from flask import Flask, request, jsonify
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModel, Trainer, TrainingArguments
from torch import nn
from torch.utils.data import DataLoader

# Initialize Flask App
app = Flask(__name__)
local_model_path = "./model_assets/base_model"


print("Downloading model Qwen 2.5")
model_id = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

print(f"Model saved to {local_model_path}")

# --- Configuration ---
MODEL_ID = "./model_assets/base_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Warning: Tokenizer load failed: {e}")
    tokenizer = None

# --- Model Definition (Fixed Forward Pass) ---
class QwenMultiHeadClassifier(nn.Module):
    def __init__(self, model_id, num_type_labels, num_code_labels, lora_config=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.qwen = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=None
        )
        # Note: LoRA config application is skipped here for inference simplicity,
        # but you would load the adapter weights in a real deployment.


        self.type_head = nn.Linear(self.config.hidden_size, num_type_labels)
        self.code_head = nn.Linear(self.config.hidden_size, num_code_labels)

        self.type_head.to(self.qwen.dtype)
        self.code_head.to(self.qwen.dtype)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state

        # Extract Last Token Embedding
        if self.config.pad_token_id is None:
            # If no pad token, assume last token is the sequence end
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # Find index of last non-padding token
                # shape: [batch_size]
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(
                    last_hidden_state.device)
            else:
                sequence_lengths = -1

        # Gather the vectors at the sequence_lengths indices
        if isinstance(sequence_lengths, int) and sequence_lengths == -1:
            # Take the last token in the sequence
            pooled_output = last_hidden_state[:, -1, :]
        else:
            batch_size = last_hidden_state.shape[0]
            pooled_output = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

        # Classification Heads
        logits_type = self.type_head(pooled_output)
        logits_code = self.code_head(pooled_output)

        return logits_type, logits_code


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
    required_cols = ['merchant_group', 'merchant_name']
    # Filter to keep ONLY the required columns, ignoring everything else
    # We use intersection to be safe, or direct selection if we enforce existence
    df = df[required_cols]

    # 3. Tokenization Logic
    dataset = Dataset.from_pandas(df, preserve_index=False)

    def preprocess_function(examples):
        inputs = [f"{g} | {n}" for g, n in zip(examples["merchant_group"], examples["merchant_name"])]
        tokenized_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        # Note: For inference, we don't strictly need labels in the dataset object
        # unless we are evaluating accuracy. We include them here as the prompt
        # implies the file has them, but the prediction loop will ignore them.
        return tokenized_inputs

    # Remove text columns to leave only tensors
    remove_cols = dataset.column_names
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)

    # Set format for PyTorch
    # We only need input_ids and attention_mask for inference
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset


# --- Helper: Model Loading (Placeholder) ---
def load_model_for_company(company, dataset):

    print(f"Loading model for {company}...")
    peft_config = LoraConfig(
        #  treat the Qwen backbone as a feature extractor.as we have custom multihead QWEN ,
        # normal classification this would have been SEQ_CLS
        # classification heads are external to the PEFT wrapper here
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,  # 16 RANk is good , LORA Mattrices will be A X R and R X B .
        lora_alpha=16,
        # Scales Output of Lora adapter by Alpha / Rank . ( 32/16 for us) , Makes learnt weights LOUDER compared to base model weights .
        # Scale of 2 is good .
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # Q K V and Output Projections selected to train as part of adapter
        # MLP Layers are gate_proj, up_proj, down_proj , but results in very large number of paramters to learn but also gives huge accuracy benefit .
        lora_dropout=0.05  # Prevents overfittig whern data sizes are small like our company data case.
    )
    # Dummy dimensions (replace with actual logic to fetch metadata for the company)
    num_type_labels = 10
    num_code_labels = 20
    model = QwenMultiHeadClassifier(
        model_id=MODEL_ID,
        num_type_labels=2,  # Dummy
        num_code_labels=2  # Dummy
    )
    model.qwen.resize_token_embeddings(len(tokenizer))

    # CRITICAL FIX 2: Wrap in PEFT immediately
    # This adds the "base_model.model..." structure so load_adapter works
    model.qwen = get_peft_model(model.qwen, peft_config)

    # Move to GPU once
    model.to("cuda")
    print(f"QWEn Multihead with 2 linear heads :\n {model}")
    adapter_path = f"./final_adapters_QWEN/{company}"
    adapter_name = f"adapter_{company}"
    try:
        # Load state dicts to cpu to inspect shapes
        type_state = torch.load(os.path.join(adapter_path, "type_head.bin"), map_location="cpu")
        code_state = torch.load(os.path.join(adapter_path, "code_head.bin"), map_location="cpu")

        n_types = type_state['weight'].shape[0]
        n_codes = code_state['weight'].shape[0]

        # Resize the linear layers on the model
        # We reuse the existing dtype/device
        dtype = model.qwen.dtype
        device = model.qwen.device

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
        model.qwen.load_adapter(adapter_path, adapter_name)
        model.qwen.set_adapter(adapter_name)
        print("MODEL READY")
        return model
    except Exception as ex:
        print(f"Skipping Global Adapter Loading : Adapter load failed. {ex}")
        return None

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        return logits
    return logits


import numpy as np  # Make sure this is imported at the top


@app.route('/predict', methods=['POST'])
def predict():
    """
    Input JSON: { "company": "CompanyA", "data_file_path": "./data/test.csv" }
    Output JSON: { "company": "CompanyA", "predictions": [ {"merchant_group": "...", "merchant_name": "...", "cc_type_id": 1, "cc_code_id": 5}, ... ] }
    """
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
