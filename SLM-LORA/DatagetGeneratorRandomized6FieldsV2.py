import pandas as pd
import random
import string


def generate_multi_company_dataset_updated(companies: list, num_records_per_company: int = 2000, seed: int = 42):
    rng = random.Random(seed)

    # Pre-define some fake company names
    fake_merchant_names = [f"Merchant_{i}" for i in range(5000)]

    all_data = []

    for company_id in companies:
        print(f"Generating data for {company_id}...")

        # --- 1. Company-Specific Target Space ---
        # Each company has its own set of 20 GL Codes and 20 Transaction Types
        # Format: COMP_A-123456 (Guarantees no overlap between companies)

        # Generate 20 unique codes
        codes = set()
        while len(codes) < 20:
            c = f"{company_id}-{rng.randint(100000, 999999)}"
            codes.add(c)
        codes = list(codes)

        # Generate 20 unique types
        types = set()
        suffix = company_id.split('_')[-1]  # "COMPANY_A" -> "A"
        base_types = ['POS', 'ONLINE', 'FEE', 'BILL', 'TRANSFER', 'WITHDRAWAL', 'DEPOSIT', 'REFUND', 'ADJUSTMENT',
                      'INTEREST']
        while len(types) < 20:
            # Create variations like POS_1_A, POS_2_A to reach 20
            base = rng.choice(base_types)
            # Add a random digit to base to ensure we can get 20 distinct types
            variation = rng.randint(1, 99)
            t = f"{base}_{variation}_{suffix}"
            types.add(t)
        types = list(types)

        # Create valid (Type, Code) pairs?
        # The prompt says "combination of 20 codes and 20 types".
        # We can either pair them 1-to-1 (20 pairs) or allow full cross-product.
        # Given the previous context of "200 combinations", let's create a pool of valid target pairs.
        # To make it "learnable", we should probably have specific valid pairs.
        # Let's create 40 valid pairs from these 20 codes and 20 types to give some variety,
        # or just random pairs.

        # Let's stick to the previous logic: A fixed set of targets.
        # If we have 20 codes and 20 types, we can form pairs.
        targets = []
        for _ in range(40):  # Keep 40 target pairs derived from the 20/20 pools
            c = rng.choice(codes)
            t = rng.choice(types)
            targets.append((t, c))

        # --- 2. Company-Specific Merchant Profiles (The Rules) ---
        # Each company deals with a unique set of 200 merchants.

        company_merchants = rng.sample(fake_merchant_names, k=200)

        profiles = []
        for merchant_name in company_merchants:
            distinct_name = merchant_name + " " + ''.join(rng.choices(string.ascii_uppercase, k=3))

            group = rng.choice(['Retail', 'Travel', 'Food', 'Tech', 'Utilities', 'Logistics', 'Consulting'])
            category = rng.choice(['Store', 'Airline', 'Cafe', 'SaaS', 'Electric', 'Hotel', 'Uber'])
            currency = rng.choice(['USD', 'EUR', 'INR', 'GBP'])

            # ASSIGN RULE: This merchant maps to ONE specific target pair from our pool
            target = rng.choice(targets)

            profiles.append({
                "merchant_name": distinct_name,
                "merchant_group": group,
                "merchant_category": category,
                "currency": currency,
                "cc_type": target[0],  # Truth
                "cc_code": target[1]  # Truth
            })

        # --- 3. Generate Records ---
        for _ in range(num_records_per_company):
            profile = rng.choice(profiles)
            amount = round(rng.uniform(10, 5000), 2)
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
df_multi = generate_multi_company_dataset_updated(companies_list, num_records_per_company=2000)

# Save individual files
for comp in companies_list:
    subset = df_multi[df_multi['company_id'] == comp]
    filename = f"{comp.lower()}_data_v2.csv"
    subset.to_csv(filename, index=False)
    print(f"Saved {filename} ({len(subset)} rows)")

    # Verify uniqueness constraint
    u_codes = subset['cc_code'].nunique()
    u_types = subset['cc_type'].nunique()
    print(f"  - Unique Codes: {u_codes} (Expected ~20)")
    print(f"  - Unique Types: {u_types} (Expected ~20)")
