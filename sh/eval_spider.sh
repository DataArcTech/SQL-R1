echo "Evaluating Spider dataset..."

PRED_SQL=
MODE=test # [dev, test]
POST_PROCESS_MODE=Maj


if [ "$MODE" = "dev" ]; then
    GOLD_SQL=data/NL2SQL/Spider/dev_gold.sql
    DB=data/NL2SQL/Spider/database
    TABLE=data/NL2SQL/Spider/tables.json
elif [ "$MODE" = "test" ]; then
    GOLD_SQL=data/NL2SQL/Spider/test_gold.sql
    DB=data/NL2SQL/Spider/test_database
    TABLE=data/NL2SQL/Spider/test_tables.json
else
    echo "Only support dev or test mode for Spider"
    exit 1
fi

ETYPE=all
PLUG_VALUE=false
KEEP_DISTINCT=false
PROGRESS_BAR_FOR_EACH_DATAPOINT=false
SAVE_DIR=results/eval/spider


if [ "$POST_PROCESS_MODE" = "Maj" ]; then
    python src/evaluation_spider_post.py \
        --pred $PRED_SQL \
        --gold $GOLD_SQL \
        --db_path $DB/ \
        --table $TABLE \
        --mode major_voting \
        --save_pred_sqls False \
        --save_dir $SAVE_DIR

    PRED_SQL=${PRED_SQL%.*}_pred_major_voting_sqls.txt
    python src/evaluation_spider.py \
        --gold_sql $GOLD_SQL \
        --pred_sql $PRED_SQL \
        --db $DB \
        --table $TABLE \
        --etype $ETYPE \
        --plug_value $PLUG_VALUE \
        --keep_distinct $KEEP_DISTINCT \
        --progress_bar_for_each_datapoint $PROGRESS_BAR_FOR_EACH_DATAPOINT \
        --save_dir $SAVE_DIR

elif [ "$POST_PROCESS_MODE" = "Gre" ]; then
    python src/evaluation_spider_post.py \
        --pred $PRED_SQL \
        --gold $GOLD_SQL \
        --db_path $DB/ \
        --table $TABLE \
        --mode greedy_search \
        --save_pred_sqls False \
        --save_dir $SAVE_DIR

    PRED_SQL=${PRED_SQL%.*}_pred_greedy_search_sqls.txt
    python src/evaluation_spider.py \
        --gold_sql $GOLD_SQL \
        --pred_sql $PRED_SQL \
        --db $DB \
        --table $TABLE \

else
    echo 'Please set the post-processing mode'
fi