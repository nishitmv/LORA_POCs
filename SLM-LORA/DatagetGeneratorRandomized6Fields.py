import pandas as pd
import random
import string


def generate_multi_company_dataset(companies: list, num_records_per_company: int = 10000, seed: int = 42):
    rng = random.Random(seed)

    # Pre-define some fake company names
    # We create a large pool so different companies pick different merchants
    fake_merchant_names = [f"Merchant_{i}" for i in range(5000)]

    all_data = []

    for company_id in companies:
        print(f"Generating data for {company_id}...")

        # --- 1. Company-Specific Target Space ---
        # Each company has its own set of 40 GL Codes and Transaction Types
        # Format: COMP_A-123456 (Guarantees no overlap between companies)
        targets = []
        for _ in range(40):
            code = f"{company_id}-{rng.randint(100000, 999999)}"
            # Type suffix matches company ID to ensure uniqueness
            suffix = company_id.split('_')[-1]  # "COMPANY_A" -> "A"
            ctype = rng.choice(['POS', 'ONLINE', 'FEE', 'BILL', 'TRANSFER']) + f"_{suffix}"
            targets.append((ctype, code))

        # --- 2. Company-Specific Merchant Profiles (The Rules) ---
        # Each company deals with a unique set of 200 merchants.
        # We enforce this by sampling from the fake_merchant_names pool without replacement per company logic
        # (or just pick random ones, collision probability is low, but let's be safe)

        company_merchants = rng.sample(fake_merchant_names, k=200)

        profiles = []
        for merchant_name in company_merchants:
            # Generate static attributes for this merchant
            # Unique suffix ensures even if "Merchant_1" appears in Company B, it's distinct like "Merchant_1 ABC"
            distinct_name = merchant_name + " " + ''.join(rng.choices(string.ascii_uppercase, k=3))

            group = rng.choice(['Retail', 'Travel', 'Food', 'Tech', 'Utilities', 'Logistics', 'Consulting'])
            category = rng.choice(['Store', 'Airline', 'Cafe', 'SaaS', 'Electric', 'Hotel', 'Uber'])
            currency = rng.choice(['USD', 'EUR', 'INR', 'GBP'])

            # ASSIGN RULE: This merchant maps to ONE specific target pair
            # This makes the pattern learnable: Name -> Code
            target = rng.choice(targets)

            profiles.append({
                "merchant_name": distinct_name,
                "merchant_group": group,
                "merchant_category": category,
                "currency": currency,
                "cc_type": target[0],  # Truth
                "cc_code": target[1]  # Truth
            })

        # --- 3. Generate Records (Noise & Variation) ---
        for _ in range(num_records_per_company):
            # Pick a rule (profile)
            profile = rng.choice(profiles)

            # Vary the amount (Model must learn Amount is NOT the signal)
            amount = round(rng.uniform(10, 5000), 2)

            # Vary description (Signal + Noise)
            # "TXN Merchant_Name_ABC REF-12345"
            noise = ''.join(rng.choices(string.digits + string.ascii_uppercase, k=8))
            description = f"TXN {profile['merchant_name']} REF-{noise}"

            row = profile.copy()
            row['company_id'] = company_id
            row['amount'] = amount
            row['description'] = description

            all_data.append(row)

    return pd.DataFrame(all_data)


# Generate for 5 companies
companies_list = ["COMPANY_A", "COMPANY_B", "COMPANY_C", "COMPANY_D", "COMPANY_E"]
df_multi = generate_multi_company_dataset(companies_list, num_records_per_company=2000)

# Save individual files
for comp in companies_list:
    subset = df_multi[df_multi['company_id'] == comp]
    filename = f"{comp.lower()}_data.csv"
    subset.to_csv(filename, index=False)
    print(f"Saved {filename} ({len(subset)} rows)")

# Verification
print("\n--- Verification ---")
print(f"Total Rows: {len(df_multi)}")
print(f"Unique GL Codes per Company: {df_multi.groupby('company_id')['cc_code'].nunique()}")

# Check overlap (Should be 0)
code_overlap = df_multi.groupby('cc_code')['company_id'].nunique()
print(f"Codes appearing in >1 company: {len(code_overlap[code_overlap > 1])}")