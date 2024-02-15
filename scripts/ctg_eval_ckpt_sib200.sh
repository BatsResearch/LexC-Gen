# TODO: change total to 200.

CKPT_DIR=/users/zyong2/data/zyong2/scaling/data/processed/905-lexcgen-pub/bloomz-7b1-sib200-en-ctg/
TGT_LANG="gn"
# for all checkpoints in CKPT_DIR
for CKPT in ${CKPT_DIR}/checkpoint-*; do
    python3 ./scripts/ctg_eval_ckpt.py \
        --peft_model_id $CKPT \
        --lexicons_dir "/oscar/data/sbach/zyong2/scaling/data/external/url-nlp/gatitos" \
        --task_data "sib200" \
        --tgt_lang $TGT_LANG \
        --total 200 \
        --top_p 0.1 \
        --temperature 1.0
done