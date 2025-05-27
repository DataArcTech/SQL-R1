import json
import argparse
import os
import random
import re

from evaluation_bird_post import major_voting, mark_invalid_sqls

random.seed(42)

def format_sql(sql):
    sql = sql.strip()
    # remove multi-line comments /* ... */
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # remove single-line comments --
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)

    sql = sql.replace("\n", " ").replace("\t", " ")
    sql = sql.strip()
    
    if sql == "":
        sql = "Error SQL"

    return sql
    
def run_spider_eval(gold_file, pred_file, db_path, table, mode, save_pred_sqls, save_dir):
    # assert mode in ["greedy_search", "major_voting"]
    gold_sqls = [line.split("\t")[0].strip() for line in open(gold_file).readlines()]
    db_ids = [line.split("\t")[1].strip() for line in open(gold_file).readlines()]
    pred = json.load(open(pred_file))
    pred_sql_key = "pred_sqls"
    # pred_sql_key = "responses"

    pred_sqls = []
    if mode == "greedy_search":
        pred_sqls = [pred_data[pred_sql_key][0] for pred_data in pred]
        assert len(pred_sqls) == len(db_ids)
        db_files = [os.path.join(db_path, db_id, db_id + ".sqlite") for db_id in db_ids]
        pred_sqls = mark_invalid_sqls(db_files, pred_sqls)
    elif mode == "major_voting":
        # perform major voting using the BIRD's evaluation script
        sampling_num = len(pred[0][pred_sql_key])
        print("sampling_num:", sampling_num)

        all_db_files = []
        for db_id in db_ids:
            all_db_files.extend([os.path.join(db_path, db_id, db_id + ".sqlite")] * sampling_num)

        all_pred_sqls = []
        for pred_data in pred:
            all_pred_sqls.extend(pred_data[pred_sql_key])
        assert len(all_db_files) == len(all_pred_sqls)

        pred_sqls = major_voting(all_db_files, all_pred_sqls, sampling_num, False)

    pred_sqls = [format_sql(pred_sql) for pred_sql in pred_sqls]
    assert len(pred_sqls) == len(gold_sqls)
    
    if save_pred_sqls:
        with open(pred_file[:-5] + f"_pred_{mode}_sqls.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(pred_sqls, indent=2 ,ensure_ascii=False))

    new_txt_file_name = pred_file.rsplit('.', 1)[0]
    with open(new_txt_file_name + f"_pred_{mode}_sqls.txt", 'w') as temp_file:
        for pred_sql in pred_sqls:
            temp_file.write(pred_sql + "\n")
        temp_file_name = temp_file.name
        print(temp_file_name)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type = str, default = "predict_dev.json")
    parser.add_argument('--gold', type = str, default = "./data/spider/dev_gold.sql")
    parser.add_argument('--db_path', type = str, default = "./data/spider/database/")
    parser.add_argument('--table', type = str, default = "./data/spider/tables.json")
    parser.add_argument('--mode', type = str, default = "greedy_search")
    parser.add_argument('--save_pred_sqls', type = bool, default = False)
    parser.add_argument('--save_dir', type = str, default = "results/eval/spider")
    opt = parser.parse_args()

    run_spider_eval(opt.gold, opt.pred, opt.db_path, opt.table, opt.mode, opt.save_pred_sqls, opt.save_dir)