echo "Evaluating Bird dataset..."

POST_PROCESS_MODE=Maj # [Maj]

PRED_SQL_PATH=

if [ "$POST_PROCESS_MODE" = "None" ]; then
    PRED_SQL_JSON_PATH=${PRED_SQL_PATH%.*}.json
else
    echo "Current post-processing mode is $POST_PROCESS_MODE"
fi

GROUND_TRUTH_SQL_PATH=data/BIRD/dev.sql
GROUND_TRUTH_JSON_PATH=data/BIRD/dev.json
DB_ROOT_PATH=data/BIRD/dev_databases/
NUM_CPUS=64
META_TIME_OUT=30.0
MODE_GT=gt
MODE_PREDICT=gpt
ITERATE_NUM=100
DIFF_JSON_PATH=data/BIRD/dev.json
SAVE_DIR=results/eval/bird
EVAL_MODE=acc


if [ "$POST_PROCESS_MODE" = "Gre" ]; then
    python src/evaluation_bird_post.py \
        --pred $PRED_SQL_PATH \
        --gold $GROUND_TRUTH_JSON_PATH \
        --db_path $DB_ROOT_PATH \
        --mode greedy_search

elif [ "$POST_PROCESS_MODE" = "Maj" ]; then
    python src/evaluation_bird_post.py \
        --pred $PRED_SQL_PATH \
        --gold $GROUND_TRUTH_JSON_PATH \
        --db_path $DB_ROOT_PATH \
        --mode major_voting
else
    echo 'Please set the post-processing mode'
fi