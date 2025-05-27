import re
import os
import sys
sys.path.append('..')
from typing import Dict, Tuple, Optional
from func_timeout import func_timeout, FunctionTimedOut

from .exec_eval import eval_exec_match
import signal

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    think_pattern = r'<think>(.*?)</think>'
    think_matches = list(re.finditer(think_pattern, processed_str, re.DOTALL))

    if not think_matches:
        print("[Error] No valid think tags found")
        final_think = None
    else:
        final_think = think_matches[-1].group(1).strip()
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, final_think, processed_str
        
    final_answer = matches[-1].group(1).strip()

    return final_answer, final_think, processed_str

def parse_sql_from_answer(answer_text: str) -> Optional[str]:
    """Parses SQL from the model's answer text.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        
    Returns:
        SQL string, or None if no SQL is found
    """
    sql_pattern = r'```sql(.*?)```'
    matches = list(re.finditer(sql_pattern, answer_text, re.DOTALL))
    
    if not matches:
        print("[Error] No valid SQL tags found")
        return None
    
    print(f"[Parsed SQL]: {matches[-1].group(1).strip()}")
    return matches[-1].group(1).strip()

def validate_response_structure(answer_str: str, processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("Tag sequence validation passed")

    # Extract SQL from answer text
    if validation_passed:
        pred_sql = parse_sql_from_answer(answer_str)
        if not pred_sql:
            validation_passed = False
    else:
        pred_sql = None

    return pred_sql, validation_passed

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
    Returns:
        Total score (sum of format and answer rewards)
    """
    FORMAT_REWARD = 1
    EXEC_REWARD = 2
    RESULT_REWARD = 3

    LIMIT_LENGTH = 2048

    total_score = 0
    print("\n" + "="*80)
    print(" Processing New NL2SQL Sample ".center(80, '='))

    # Parse ground truth data
    db_name = ground_truth.get('db_id', '').replace('\n', '').strip()
    gold_sql = re.sub(r'\s+', ' ', ground_truth.get('sql', ''))
    # Extract model answer
    answer_text, think_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model's Response] {processed_str}")

    # Format Reward
    pred_sql, format_correct = validate_response_structure(answer_text, processed_str)
    format_score = FORMAT_REWARD if format_correct else -abs(FORMAT_REWARD)
    print(f"\n[Format validation] {'PASS' if format_correct else 'FAIL'}")
    print(f"[Format score]: {format_score}")

    db_path = os.path.join('data/NL2SQL/SynSQL-2.5M/databases', db_name, db_name + '.sqlite')

    exec_score = 0
    result_score = 0
    if format_correct and pred_sql:
        # Validate Exec Score
        pred_sql = re.sub(r'\s+', ' ', pred_sql)
        print(f"[DB NAME]: {db_name}")
        print(f"[Gold SQL]: {gold_sql}")
        print(f"[Pred SQL]: {pred_sql}")


        try:
            exec_status = func_timeout(
                timeout=30,
                func=eval_exec_match,
                args=(db_path, pred_sql, gold_sql),
                kwargs={
                    'plug_value': False, 
                    'keep_distinct': False, 
                    'progress_bar_for_each_datapoint': False
                }
            )
        except FunctionTimedOut:
            exec_status = 'Unexecutable'
        print(f"[Exec status]: {exec_status}")

        if exec_status == 'Unexecutable':
            exec_score = -abs(EXEC_REWARD)
            result_score = 0
        elif exec_status == 'Gold Error':
            exec_score = 0
            result_score = 0        
        elif exec_status == 'Mismatch':
            exec_score = EXEC_REWARD
            result_score = -abs(RESULT_REWARD)
        elif exec_status == 'Match':
            exec_score = EXEC_REWARD
            result_score = RESULT_REWARD

    # Length Reward v1: 鼓励输出接近 LIMIT_LENGTH，同时增加 SQL 在 answer 中的比例，但是要在 SQL 可执行才有意义
    # Length Reward v2: 更严格的长度奖励，只有 match 才有分，且比例更小 1 分
    # if format_correct and (exec_status == 'Mismatch' or exec_status == 'Match'):
    if format_correct and exec_status == 'Match':
        pos_length = len(think_text) + len(answer_text)
        if pos_length <= LIMIT_LENGTH:
            sql_in_answer_sub_score = len(pred_sql) / len(answer_text)
            length_sub_score = pos_length / LIMIT_LENGTH * 0.5
            length_score = length_sub_score + sql_in_answer_sub_score
            print(f"[Length pos_length]: {pos_length}")
            print(f"[Length pred_sql]: {len(pred_sql)}")
            print(f"[Length answer_text]: {len(answer_text)}")
        else:
            sql_in_answer_sub_score = len(pred_sql) / len(answer_text)
            length_score = 0.5 + sql_in_answer_sub_score
    else:
        length_score = 0

    

    total_score = format_score + exec_score + result_score + length_score

    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f" -- Format Score: {format_score}")
    print(f" -- Exec Score: {exec_score}")
    print(f" -- Result Score: {result_score}")
    print(f" -- Length Score: {length_score}")
    print(f" -- Total Score: {total_score}")
    print("="*80 + "\n")

    return total_score
