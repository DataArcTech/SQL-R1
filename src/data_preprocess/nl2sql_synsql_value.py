""" Preprocess dataset for knights and knaves logic task """

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import sqlite3

from src.utils.prepare_input_seq import get_input_seq


def make_prefix(dp, template_type, db_path):
    
    if template_type == 'base':
        prefix = """The user asks a question about a database, and the Assistant helps convert it to SQL.
The assistant first thinks about how to write the SQL query by analyzing the question, database schema and external knowledge, then provides the final SQL query. 
The reasoning process and SQL query are enclosed within <think> </think> and <answer> </answer> tags respectively.
The answer must contain the SQL query within '''sql ''' tags.

Database Schema:
{schema}

External Knowledge:
{external_knowledge}

SQL Complexity Level: {sql_complexity}

For example:
<think>
To translate the given natural language question into an executable SQLite query, we need to follow these detailed steps:
1. **Identify Key Elements**: The question queries for code snippets that are both complicated (complexity score > 5) and public (`is_public` = 1). We need to retrieve their descriptions and complexity scores.
2. **Focus on Relevant Tables**: The `code_snippets` table contains the necessary fields (`description`, `complexity`, `is_public`).
3. **Construct the Query**: We should select the required fields (`description` and `complexity`) from the `code_snippets` table. We also apply the conditions specified in the question to filter the results.
4. **Ordering**: The reference solution includes an `ORDER BY` clause to sort results by complexity in descending order, which is a reasonable way to present the data to highlight the most complex snippets first.
5. **Final Query Construction**: Putting all this together into a SQL query.
</think>
<answer>
Here's how the query can be written:
```sql
SELECT description, complexity FROM code_snippets WHERE complexity > 5 AND is_public = 1 ORDER BY complexity DESC;
```
This query retrieves the descriptions and complexity scores of code snippets that are both complicated (complexity > 5) and publicly available (`is_public` = 1), sorted by complexity in descending order.
This solution is straightforward and precisely matches the requirements of the question. It avoids unnecessary complexities, such as joining or selecting columns not relevant to the query itself.
</answer>

User: {question}
Assistant: <think>"""
        
    elif template_type == 'qwen-instruct':
        prefix = """<|im_start|>system
You are a helpful SQL expert assistant.
The assistant first thinks about how to write the SQL query by analyzing the question, database schema and external knowledge, then provides the final SQL query.
The reasoning process and SQL query are enclosed within <think> </think> and <answer> </answer> tags respectively.
The answer must contain the SQL query within ```sql ``` tags.

Database Schema:
{schema}

External Knowledge: {external_knowledge}

SQL Complexity Level: {sql_complexity}

For example:
<think>
To translate the given natural language question into an executable SQLite query, we need to follow these detailed steps:
1. **Identify Key Elements**: The question queries for code snippets that are both complicated (complexity score > 5) and public (`is_public` = 1). We need to retrieve their descriptions and complexity scores.
2. **Focus on Relevant Tables**: The `code_snippets` table contains the necessary fields (`description`, `complexity`, `is_public`).
3. **Construct the Query**: We should select the required fields (`description` and `complexity`) from the `code_snippets` table. We also apply the conditions specified in the question to filter the results.
4. **Ordering**: The reference solution includes an `ORDER BY` clause to sort results by complexity in descending order, which is a reasonable way to present the data to highlight the most complex snippets first.
5. **Final Query Construction**: Putting all this together into a SQL query.
</think>
<answer>
Here's how the query can be written:
```sql
SELECT description, complexity FROM code_snippets WHERE complexity > 5 AND is_public = 1 ORDER BY complexity DESC;
```
This query retrieves the descriptions and complexity scores of code snippets that are both complicated (complexity > 5) and publicly available (`is_public` = 1), sorted by complexity in descending order.
This solution is straightforward and precisely matches the requirements of the question. It avoids unnecessary complexities, such as joining or selecting columns not relevant to the query itself.
</answer>
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>
"""
        
    prefix = get_input_seq(
        data=dp,
        database_path=db_path,
        dataset_name='synsql',
        table_value_cache_path='data/NL2SQL/SynSQL-2.5M/synsql_db_id2sampled_db_values.json',
        table_info_cache_path='data/NL2SQL/SynSQL-2.5M/synsql_db_id2db_info.json',
        input_prompt=prefix
    )
        
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/NL2SQL/SynSQL-2.5M/segmentation/5000/instruct/sampled_Highly_Complex')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='data/NL2SQL/SynSQL-2.5M/segmentation/5000/sampled_Highly_Complex.jsonl')
    parser.add_argument('--train_size', type=int, default=4000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    parser.add_argument('--db_path', default='data/NL2SQL/SynSQL-2.5M/databases')
    
    args = parser.parse_args()
    
    data_source = 'synsql'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type, db_path=args.db_path)
            solution = {
                "db_id": example['db_id'],
                "sql": example['sql']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)