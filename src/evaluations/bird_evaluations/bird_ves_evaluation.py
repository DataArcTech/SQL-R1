import os
import pdb
import sys
import json
import numpy as np
import argparse
import sqlite3
import multiprocessing as mp
import yaml
from func_timeout import func_timeout, FunctionTimedOut
import time
import math
from tqdm import tqdm

def result_callback(result):
    exec_result.append(result)

def clean_abnormal(input):
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input,axis=0)
    std = np.std(input,axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list

def execute_sql(sql, db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    start_time = time.time()
    cursor.execute(sql)
    exec_time = time.time() - start_time
    return exec_time

def iterated_execute_sql(predicted_sql,ground_truth,db_path,iterate_num):
    conn = sqlite3.connect(db_path)
    diff_list = []
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    time_ratio = 0
    if set(predicted_res) == set(ground_truth_res):
        for i in range(iterate_num):
            predicted_time = execute_sql(predicted_sql, db_path)
            ground_truth_time = execute_sql(ground_truth, db_path)
            diff_list.append(ground_truth_time / predicted_time)
        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list) / len(processed_diff_list)
    return time_ratio



def execute_model(predicted_sql,ground_truth, db_place, idx, iterate_num, meta_time_out):
    try:
        # you can personalize the total timeout number
        # larger timeout leads to more stable ves
        # while it needs more your patience....
        time_ratio = func_timeout(meta_time_out * iterate_num, iterated_execute_sql,
                                  args=(predicted_sql, ground_truth, db_place, iterate_num))
        # print([idx, math.sqrt(time_ratio)])
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        time_ratio = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        time_ratio = 0
    result = {'sql_idx': idx, 'time_ratio': time_ratio}
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path, 'r'))
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split('\t----- bird -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(sql_path)
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, iterate_num=100, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, iterate_num, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_ves(exec_results):
    num_queries = len(exec_results)
    total_ratio = 0
    count = 0

    for i, result in enumerate(exec_results):
        if result['time_ratio'] != 0:
            count += 1
        total_ratio += math.sqrt(result['time_ratio']) * 100
    ves = (total_ratio/num_queries)
    return ves

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents

def compute_ves_by_diff(exec_results,diff_json_path):
    num_queries = len(exec_results)
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []
    for i,content in enumerate(contents):
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])
        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])
        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])
    simple_ves = compute_ves(simple_results)
    moderate_ves = compute_ves(moderate_results)
    challenging_ves = compute_ves(challenging_results)
    all_ves = compute_ves(exec_results)
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_ves, moderate_ves, challenging_ves, all_ves, count_lists

def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('=========================================    VES   ========================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('ves', *score_lists))

def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path='config/evaluation_ves_config.yaml'):
    """主函数入口"""
    # 加载配置
    config = load_config(config_path)
    
    # 获取评估参数
    predicted_sql_path = config['predicted_sql_path']
    ground_truth_path = config['ground_truth_path']
    data_mode = config['data_mode']
    db_root_path = config['db_root_path']
    num_cpus = config['num_cpus']
    meta_time_out = config['meta_time_out']
    mode_gt = config['mode_gt']
    mode_predict = config['mode_predict']
    iterate_num = config['iterate_num']
    diff_json_path = config['diff_json_path']
    save_dir = config['save_dir']

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化结果列表
    global exec_result
    exec_result = []

    # 处理预测SQL
    pred_queries, db_paths = package_sqls(predicted_sql_path, db_root_path, 
                                        mode=mode_predict,
                                        data_mode=data_mode)
    
    # 处理真实SQL
    gt_queries, db_paths_gt = package_sqls(ground_truth_path, db_root_path, 
                                         mode='gt',
                                         data_mode=data_mode)

    # 执行评估
    query_pairs = list(zip(pred_queries, gt_queries))
    
    print('开始执行VES评估...')
    run_sqls_parallel(query_pairs, db_places=db_paths, 
                     num_cpus=num_cpus,
                     iterate_num=iterate_num,
                     meta_time_out=meta_time_out)
    
    # 排序结果
    exec_result = sort_results(exec_result)
    
    print('开始计算VES指标...')
    simple_ves, moderate_ves, challenging_ves, ves, count_lists = \
        compute_ves_by_diff(exec_result, diff_json_path)
    
    # 打印结果
    score_lists = [simple_ves, moderate_ves, challenging_ves, ves]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("评估完成")

    # 保存结果
    results = {
        'simple_ves': simple_ves,
        'moderate_ves': moderate_ves,
        'challenging_ves': challenging_ves,
        'overall_ves': ves,
        'counts': {
            'simple': count_lists[0],
            'moderate': count_lists[1],
            'challenging': count_lists[2],
            'total': count_lists[3]
        }
    }

    name = predicted_sql_path.split('/')[-1].split('.')[0]
    
    # 保存结果到JSON文件
    result_file = os.path.join(save_dir, name + '_ves_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/evaluation_ves_config.yaml',
                      help='配置文件路径')
    args = parser.parse_args()
    
    main(args.config)


