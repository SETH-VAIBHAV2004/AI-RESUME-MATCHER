# download_datasets.py
from datasets import load_dataset
import pandas as pd
import os
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

def load_and_flatten_dataset(dataset_name, split="train", filename="output.csv"):
    """Loads a dataset safely, flattens nested fields, and saves as CSV."""
    try:
        print(f"\n⬇️ Downloading {dataset_name} ...")
        ds = load_dataset(dataset_name, split=split, streaming=True)
        records = []

        for example in tqdm(ds, desc=f"Processing {dataset_name}"):
            flat_example = {}
            for k, v in example.items():
                # Flatten nested dictionaries if needed
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat_example[f"{k}_{sub_k}"] = sub_v
                else:
                    flat_example[k] = v
            records.append(flat_example)

        # Convert to DataFrame
        df = pd.DataFrame(records)
        df.to_csv(filename, index=False)
        print(f"✅ Saved {dataset_name} → {len(df)} rows to {filename}")

    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")

# --- Dataset 1 ---
load_and_flatten_dataset("netsol/resume-score-details", filename="data/resume_score_details.csv")

# --- Dataset 2 ---
load_and_flatten_dataset("cnamuangtoun/resume-job-description-fit", filename="data/resume_job_fit.csv")

# --- Dataset 3 ---
load_and_flatten_dataset("facehuggerapoorv/resume-jd-match", filename="data/resume_jd_match.csv")

# --- Dataset 4 (optional) ---
load_and_flatten_dataset("azrai99/job-dataset", filename="data/job_dataset.csv")

print("\n🎯 All datasets downloaded and saved successfully!")
