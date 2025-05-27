import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
import yaml
import os

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res



def execute_model(predicted_sql,ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0
    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'res': res}
    # print(result)
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
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results,diff_json_path):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i,content in enumerate(contents):
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])

        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])

        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])

    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results)
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results)
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results)
    all_acc = sum(results)/num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists



def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path='config/evaluation_config.yaml'):
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
    difficulty = config['difficulty']
    diff_json_path = config['diff_json_path']
    save_dir = config['save_dir']

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
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
    run_sqls_parallel(query_pairs, db_places=db_paths, 
                     num_cpus=num_cpus, 
                     meta_time_out=meta_time_out)
    
    # 排序结果
    exec_result = sort_results(exec_result)
    
    print('开始计算指标...')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_result, diff_json_path)
    
    # 打印结果
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("评估完成")

    # 保存结果
    results = {
        'simple_acc': simple_acc,
        'moderate_acc': moderate_acc,
        'challenging_acc': challenging_acc,
        'overall_acc': acc,
        'counts': {
            'simple': count_lists[0],
            'moderate': count_lists[1],
            'challenging': count_lists[2],
            'total': count_lists[3]
        }
    }

    name = predicted_sql_path.split('/')[-1].split('.')[0]
    
    # 保存结果到JSON文件
    result_file = os.path.join(save_dir, name + '_eval_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/evaluation_config.yaml',
                      help='配置文件路径')
    args = parser.parse_args()
    
    main(args.config)
    