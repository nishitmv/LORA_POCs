import pandas as pd
import random
import string
from faker import Faker


def generate_randomized_dataset(num_records=10000, num_unique_classes=30, num_merchants=500):
    """
    Generates a completely new dataset every run.
    Everything from the vocabulary (Types, Codes) to the Merchant names is randomized.
    """
    # Re-seed Faker and Random with system time (default) for total randomness
    fake = Faker()
    # Explicitly clear any previous seeds to ensure randomness
    random.seed(None)
    Faker.seed(None)

    # --- 1. Randomize the "Vocabulary" (Labels) ---

    # A. Generate Random Transaction Types (e.g., "TYPE_X92", "ACH_Payment_B")
    # We build 10-15 random types dynamically
    prefixes = ['POS', 'ACH', 'Wire', 'Card', 'Transfer', 'Fee', 'Auto']
    generated_types = set()
    while len(generated_types) < 15:
        p = random.choice(prefixes)
        s = ''.join(random.choices(string.ascii_uppercase, k=3))  # e.g. "POS_ABC"
        generated_types.add(f"{p}_{s}")
    trans_type_vocab = list(generated_types)

    # B. Generate Random GL Codes (e.g., 5-digit numbers in random ranges)
    # Pick a random starting block (e.g., 40000 or 70000)
    base_code = random.randint(10, 90) * 1000
    gl_vocab = [str(base_code + random.randint(0, 9999)) for _ in range(100)]

    # C. Create Valid Target Pairs (The 30 output classes)
    valid_label_combinations = []
    while len(valid_label_combinations) < num_unique_classes:
        combo = (random.choice(trans_type_vocab), random.choice(gl_vocab))
        if combo not in valid_label_combinations:
            valid_label_combinations.append(combo)

    print(f"Generated Vocabulary: {len(trans_type_vocab)} Types, {len(gl_vocab)} GL Codes.")
    print(f"Target Classes: {len(valid_label_combinations)} unique pairs.")

    # --- 2. Randomize the "Features" (Inputs) ---

    # A. Generate Random Merchant Groups (Industries)
    # Pick 8 random industries from a massive list or generate generic ones
    base_industries = [
        'Retail', 'Software', 'Logistics', 'Dining', 'Travel', 'Consulting',
        'Healthcare', 'Manufacturing', 'Energy', 'Education', 'Real Estate',
        'Media', 'Telecom', 'Construction', 'Finance', 'Legal'
    ]
    # Shuffle and pick a subset so every file feels different
    selected_groups = random.sample(base_industries, k=random.randint(5, 10))

    # B. Build the Registry: (Group, Name) -> (Type, Code)
    merchant_registry = {}

    for _ in range(num_merchants):
        grp = random.choice(selected_groups)

        # Generate a truly random company name
        # We append a random suffix if needed to ensure uniqueness across runs
        raw_name = fake.company()
        suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
        name = f"{raw_name} {suffix}"  # e.g. "Smith LLC xyz"

        # Assign Deterministic Label
        assigned_label = random.choice(valid_label_combinations)
        merchant_registry[(grp, name)] = assigned_label

    # --- 3. Generate Records ---
    data = []
    merchants_list = list(merchant_registry.keys())

    for _ in range(num_records):
        grp, name = random.choice(merchants_list)
        cc_type, cc_code = merchant_registry[(grp, name)]

        data.append({
            "merchant_group": grp,
            "merchant_name": name,
            "cc_type": cc_type,
            "cc_code": cc_code
        })

    df = pd.DataFrame(data)

    # Sanity Check
    consistency = df.groupby(['merchant_group', 'merchant_name'])['cc_code'].nunique().max()
    assert consistency == 1, "Consistency Rule Violated!"

    return df


# --- Execution ---
if __name__ == "__main__":
    # Run 1
    print("\n--- Generating Run A ---")
    df_a = generate_randomized_dataset()
    print(df_a.head(3))
    df_a.to_csv("company_D.csv", index=False)

    # Run 2 (Will look totally different)
    print("\n--- Generating Run B ---")
    df_b = generate_randomized_dataset()
    print(df_b.head(3))
    df_b.to_csv("company_E.csv", index=False)

    # Run 3 (Will look totally different)
    print("\n--- Generating Run C ---")
    df_b = generate_randomized_dataset()
    print(df_b.head(3))
    df_b.to_csv("company_F.csv", index=False)
