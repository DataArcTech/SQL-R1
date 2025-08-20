import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from rapidfuzz import process

# 读取 db_names.txt
with open("db_names.txt", "r", encoding="utf-8") as f:
    db_names = eval(f.read())  # db_names.txt 的内容是 Python 字符串列表

def fix_db_name(name, db_names):
    """如果 name 不在 db_names 中，则返回编辑距离最小的 db_name"""
    if name in db_names:
        return name, False
    best_match, score, _ = process.extractOne(name, db_names, score_cutoff=0)
    return best_match, True

def np_encoder(obj):
    """处理 numpy 类型，转成 Python 类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

def process_dataset(file_path, cleaned_prefix):
    df = pd.read_parquet(file_path)
    
    type_a_missing = 0
    type_b_missing = 0

    for i in tqdm(range(len(df)), desc=f"Processing {file_path}"):
        # 处理 ['db_id']
        db_id = df.at[i, "db_id"]
        fixed, changed = fix_db_name(db_id, db_names)
        if changed:
            type_a_missing += 1
            df.at[i, "db_id"] = fixed
        
        # 处理 ['reward_model']['ground_truth']['db_id']
        try:
            gt_db_id = df.at[i, "reward_model"]["ground_truth"]["db_id"]
            fixed, changed = fix_db_name(gt_db_id, db_names)
            if changed:
                type_b_missing += 1
                reward_model = df.at[i, "reward_model"]
                reward_model["ground_truth"]["db_id"] = fixed
                df.at[i, "reward_model"] = reward_model
        except Exception:
            pass

    print(f"{file_path}: TYPE A missing = {type_a_missing}, TYPE B missing = {type_b_missing}")

    # 保存 parquet
    parquet_path = f"{cleaned_prefix}.cleaned.parquet"
    df.to_parquet(parquet_path, index=False)

    # 保存 json（处理 numpy 类型）
    json_path = f"{cleaned_prefix}.cleaned.json"
    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=np_encoder)

    print(f"Saved: {parquet_path}, {json_path}")

# 处理 train 和 test
process_dataset("train.parquet", "train")
process_dataset("test.parquet", "test")
