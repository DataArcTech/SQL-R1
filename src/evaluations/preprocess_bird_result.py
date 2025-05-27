import json
import os
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_result_path', type=str)
    parser.add_argument('--json_result_path', type=str)
    parser.add_argument('--json_save_path', type=str)
    args = parser.parse_args()
    
    txt_result_path = args.txt_result_path
    json_result_path = args.json_result_path
    json_save_path = args.json_save_path
    
    with open(txt_result_path, 'r') as f:
        result_sqls = f.readlines()
        
    with open(json_result_path, 'r') as f:
        json_result = json.load(f)
    
    final_output_dict = {}

    for sql, json_data in zip(result_sqls, json_result):
        sql = sql.split('/*')[0].strip()
        final_output_dict[str(json_data['question_id'])] = sql + "\t----- bird -----\t" + json_data['db_id']

    with open(json_save_path, 'w') as f:
        json.dump(final_output_dict, f, indent=4)