import pandas as pd
import random
import string
from faker import Faker


# -----------------------------
# Helpers
# -----------------------------
def _random_cc_types(rng: random.Random, company_suffix: str, n: int = 15):
    """
    Generates distinct transaction types.
    Adds a company_suffix (e.g., '_A') to ensure types are unique across companies.
    """
    prefixes = ['POS', 'ACH', 'Wire', 'Card', 'Tfr', 'Fee', 'Auto', 'Bill', 'Dbt', 'Crd']
    out = set()
    while len(out) < n:
        p = rng.choice(prefixes)
        # Random 2-3 char suffix
        s = ''.join(rng.choices(string.ascii_uppercase, k=random.randint(2, 3)))
        # Format: POS_XYZ_A (Guarantees Company A types != Company B types)
        out.add(f"{p}_{s}_{company_suffix}")
    return list(out)


def _generate_target_pool(rng: random.Random, num_classes: int, company_id: str):
    """
    Generates a fixed pool of (cc_type, cc_code) pairs.
    Both Type and Code are namespaced to the company to ensure global uniqueness.
    """
    # Create a short suffix from company_id (e.g., "COMPANY_A" -> "A")
    comp_suffix = company_id.split('_')[-1] if '_' in company_id else company_id[:3]

    # 1. Generate Types unique to this company
    types = _random_cc_types(rng, comp_suffix, n=min(15, num_classes))

    pool = set()
    while len(pool) < num_classes:
        t = rng.choice(types)
        # 2. Generate Code unique to this company
        # Format: COMP_A-849302
        c_suffix = rng.randint(100000, 999999)
        c = f"{company_id}-{c_suffix}"
        pool.add((t, c))

    return list(pool)


def _get_description_patterns(company_id: str):
    """
    Returns a distinct list of description formats based on the company.
    """
    if "A" in company_id:
        # Style: Dashes, Standard Prefixes (INV, REF)
        return [
            "POS - {merchant} - {cat} - INV#{inv}",
            "CARD PURCH: {merchant} ({cat}) REF-{ref}",
            "DD: {merchant} / {cat} / ORD {ord}",
            "FEE: {merchant} {cat} ID:{acct}",
            "ONLINE-TXN: {merchant} - {cat}",
        ]
    elif "B" in company_id:
        # Style: Pipes, Hash signs, Short codes
        return [
            "{merchant} | {cat} | #{inv}",
            "TXN | {merchant} | {cat} | Auth:{ref}",
            "{cat} | {merchant} | {ord}",
            "E-Bill | {merchant} | {cat} | #{acct}",
            "Auto-Debit | {merchant} | {cat}",
        ]
    else:
        # Style: Verbose, Sentence-like, Spaced out
        return [
            "Payment to {merchant} for {cat} services",
            "Authorized txn at {merchant} ({cat}) ref {ref}",
            "{merchant} {cat} purchase order {ord}",
            "Recurring payment: {merchant} [{cat}]",
            "{cat} charge from {merchant} ID {acct}",
        ]


def _make_description(fake: Faker, rng: random.Random, merchant_name: str, merchant_category: str, patterns: list):
    p = rng.choice(patterns)
    return p.format(
        merchant=merchant_name.upper(),
        cat=merchant_category.upper(),
        inv=str(rng.randint(10000, 99999)),
        ref=fake.bothify(text="??####"),
        ord=fake.bothify(text="##-####"),
        acct=str(rng.randint(100, 999)),
    )


def _pick_amount(rng: random.Random):
    bucket = rng.random()
    if bucket < 0.60:
        amt = rng.uniform(10, 500)
    elif bucket < 0.90:
        amt = rng.uniform(500, 5000)
    else:
        amt = rng.uniform(5000, 50000)
    if rng.random() < 0.05:
        amt = -amt
    return round(amt, 2)


def _pick_currency(rng: random.Random):
    currencies = ["INR", "USD", "EUR", "GBP"]
    weights = [0.80, 0.10, 0.05, 0.05]
    return rng.choices(currencies, weights=weights, k=1)[0]


# -----------------------------
# Main generator
# -----------------------------
def generate_company_dataset(
        company_id: str,
        num_records: int = 10000,
        num_merchants: int = 800,
        num_classes: int = 40,
        seed: int | None = None
):
    rng = random.Random(seed)
    fake = Faker()
    if seed is not None:
        Faker.seed(seed)

    # 1. Setup Merchants (Varying Industries)
    base_industries = [
        'Retail', 'Software', 'Logistics', 'Dining', 'Travel', 'Consulting',
        'Healthcare', 'Manufacturing', 'Energy', 'Education', 'Real Estate',
        'Media', 'Telecom', 'Construction', 'Finance', 'Legal'
    ]
    # Randomly select a subset of industries for this company
    merchant_groups = rng.sample(base_industries, k=rng.randint(8, 12))

    merchant_categories = [
        "Grocery", "Fuel", "Restaurant", "FastFood", "Pharmacy", "Hospital",
        "Airlines", "Hotel", "Taxi", "Ecommerce", "Subscription", "Utilities",
        "Electronics", "Apparel", "Entertainment", "Education"
    ]

    merchants = []
    seen_merchants = set()
    while len(merchants) < num_merchants:
        grp = rng.choice(merchant_groups)
        raw = fake.company()
        # Add a random suffix to ensure merchant names are unique per company run
        suffix = ''.join(rng.choices(string.ascii_lowercase, k=3))
        name = f"{raw} {suffix}"
        key = (grp, name)
        if key not in seen_merchants:
            seen_merchants.add(key)
            merchants.append(key)

    # 2. Generate the FIXED pool of 40 targets (Unique to this company)
    valid_targets = _generate_target_pool(rng, num_classes, company_id)

    # 3. Get Description Patterns (Unique to this company)
    desc_patterns = _get_description_patterns(company_id)

    # Registry: Map (Inputs) -> (Target Class)
    combo_registry = {}

    data = []
    for _ in range(num_records):
        # Generate inputs
        merchant_group, merchant_name = rng.choice(merchants)
        merchant_category = rng.choice(merchant_categories)
        amount = _pick_amount(rng)
        currency = _pick_currency(rng)

        # Description using company-specific patterns
        description = _make_description(fake, rng, merchant_name, merchant_category, desc_patterns)

        combo_key = (
            merchant_category,
            amount,
            currency,
            description,
            merchant_name,
            merchant_group
        )

        # Assign target (consistent per input combination)
        if combo_key in combo_registry:
            cc_type, cc_code = combo_registry[combo_key]
        else:
            # Randomly assign one of the 40 allowed targets
            cc_type, cc_code = rng.choice(valid_targets)
            combo_registry[combo_key] = (cc_type, cc_code)

        data.append({
            "company_id": company_id,
            "merchant_group": merchant_group,
            "merchant_name": merchant_name,
            "merchant_category": merchant_category,
            "amount": amount,
            "currency": currency,
            "description": description,
            "cc_type": cc_type,
            "cc_code": cc_code,
        })

    df = pd.DataFrame(data)

    # Sanity Checks
    # 1. Consistency Rule
    key_cols = ["merchant_category", "amount", "currency", "description", "merchant_name", "merchant_group"]
    assert df.groupby(key_cols)["cc_code"].nunique().max() == 1, "Consistency Rule Violated!"

    return df


if __name__ == "__main__":
    companies = ["COMPANY_A", "COMPANY_B", "COMPANY_C"]

    for i, cid in enumerate(companies, start=1):
        print(f"--- Generating {cid} ---")
        df = generate_company_dataset(
            company_id=cid,
            num_records=10000,
            num_merchants=900,
            num_classes=40,
            seed=5000 + i
        )

        filename = f"{cid.lower()}_data.csv"
        df.to_csv(filename, index=False)

        # Verify Uniqueness (Preview)
        print(f"Examples of Descriptions ({cid}):")
        print(df['description'].head(2).values)
        print(f"Examples of Types ({cid}):")
        print(df['cc_type'].unique()[:3])
        print("-" * 30)
