import pandas as pd
import random

# Configuration for variability
employment_types = [
    "Corporate (Multinational)",
    "Corporate (Local LLC)",
    "Salaried (Public Sector)",
    "Salaried (Private)",
    "Self-Employed (Professional)",
    "Self-Employed (Commercial)",
    "Freelancer (Tech)",
    "Freelancer (Creative)",
    "Retired / Pensioner",
    "Expat (Remittance)",
]

vehicle_types = [
    "New Only",
    "Used (Up to 5 yrs)",
    "Used (Up to 10 yrs)",
    "Electric/Hybrid",
    "Commercial Trucks",
]

data = []

for i in range(1, 201):
    emp = random.choice(employment_types)
    veh = random.choice(vehicle_types)

    # Logic-based parameter scaling
    base_rate = 15.0
    if "Self-Employed" in emp or "Freelancer" in emp:
        base_rate += 4.0
    if "Used" in veh:
        base_rate += 2.0
    if "Electric" in veh:
        base_rate -= 1.5

    rate = round(base_rate + random.uniform(-1, 2), 2)
    tenure = random.choice([36, 48, 60, 72, 84, 96])
    dbr = random.choice([35, 40, 45, 50, 55, 60])
    min_inc = random.choice([5000, 7000, 10000, 15000, 25000, 40000])

    policy_text = (
        f"This policy applies to {emp} applicants purchasing {veh} vehicles. "
        f"The interest rate is set at {rate}% with a maximum repayment period of {tenure} months. "
        f"Applicants must not exceed a Debt Burden Ratio (DBR) of {dbr}%. "
        f"Minimum verifiable monthly income requirement is {min_inc} EGP."
    )

    # DataRobot Mandatory Column Mapping:
    # 1. 'document' -> The text to be embedded.
    # 2. 'document_file_path' -> Unique ID ending in a file extension.
    data.append(
        {
            "document": policy_text,
            "document_file_path": f"policy_{i:03}.txt",
            "policy_id": f"POL-{i:03}",
            "employment_category": emp,
            "vehicle_eligibility": veh,
            "interest_rate": rate,
            "max_tenure_months": tenure,
            "max_dbr_percent": dbr,
            "min_income_egp": min_inc,
        }
    )

df = pd.DataFrame(data)

# Ensure the mandatory columns are at the front for easy verification
df = df[
    [
        "document",
        "document_file_path",
        "policy_id",
        "employment_category",
        "min_income_egp",
    ]
]

df.to_csv("datarobot_finance_policies.csv", index=False)
print(f"Generated {len(df)} policies in 'datarobot_finance_policies.csv'")
