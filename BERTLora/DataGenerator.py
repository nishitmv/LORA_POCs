import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

# --- Configuration ---
NUM_RECORDS = 10000
NUM_UNIQUE_LABELS = 30  # Constraint: Exactly 30 unique Type/GL-Code combinations
NUM_UNIQUE_MERCHANTS = 500  # How many distinct merchants exist in the data

# 1. Define the Pool of Possible Labels (Transaction Types & GL Codes)
# Transaction Types (Nature of the movement)
transaction_types = [
    'POS Purchase', 'Online Payment', 'Recurring Subscription',
    'Wire Transfer', 'ACH Debit', 'Service Fee', 'Refund', 'Adjustment'
]

# GL Codes (General Ledger Account Numbers)
# Generating plausible expense/asset account codes (e.g., 5-series for expenses)
gl_codes = [str(random.randint(50000, 69999)) for _ in range(50)]

# Generate exactly 30 unique combinations of (Transaction Type, GL Code)
valid_label_combinations = []
while len(valid_label_combinations) < NUM_UNIQUE_LABELS:
    combo = (random.choice(transaction_types), random.choice(gl_codes))
    if combo not in valid_label_combinations:
        valid_label_combinations.append(combo)

print(f"Generated {len(valid_label_combinations)} unique Accounting Class combinations.")

# 2. Generate Unique Merchants and Assign Ground Truth
# We map (Group, Name) -> (Trans Type, GL Code)
# "Amazon Web Services" (Software) might always map to -> "Recurring Subscription" / "60100" (Software Expense)
merchant_registry = {}
merchant_groups = [
    'Software & SaaS', 'Office Supplies', 'Travel & T&E',
    'Logistics', 'Professional Services', 'Utilities', 'Marketing'
]

for _ in range(NUM_UNIQUE_MERCHANTS):
    # Generate unique merchant features
    grp = random.choice(merchant_groups)
    name = fake.company()

    # Ensure unique merchant names within the registry
    while (grp, name) in merchant_registry:
        name = fake.company()

    # Assign one of the 30 Accounting mappings to this merchant
    assigned_label = random.choice(valid_label_combinations)
    merchant_registry[(grp, name)] = assigned_label

# 3. Generate the Dataset
data = []
merchants_list = list(merchant_registry.keys())

for _ in range(NUM_RECORDS):
    # Pick a random merchant
    grp, name = random.choice(merchants_list)

    # Retrieve the deterministic GL mapping for this merchant
    trans_type, gl_code = merchant_registry[(grp, name)]

    data.append({
        "merchant_group": grp,
        "merchant_name": name,
        "cc_type": trans_type,  # Transaction Type
        "cc_code": gl_code  # GL Code
    })

# 4. Create DataFrame and Verify
df = pd.DataFrame(data)

# Validation check
unique_label_count = df.groupby(['cc_type', 'cc_code']).ngroups
consistency_check = df.groupby(['merchant_group', 'merchant_name'])['cc_code'].nunique().max()

print(f"\n--- Dataset Summary ---")
print(f"Total Records: {len(df)}")
print(f"Unique GL/Type Combinations: {unique_label_count} (Should be 30)")
print(f"Max GL Codes per Merchant: {consistency_check} (Should be 1 - strict consistency)")

print("\n--- Sample Rows (Accounting Data) ---")
print(df.head())

# Save to CSV
df.to_csv("company_C.csv", index=False)
