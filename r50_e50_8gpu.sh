EXP_DIR=output/r50_e50

if [ ! -d "output" ]; then
    mkdir output
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi

srun -p dsta \
    --job-name=detexp \
    --gres=gpu:8 \
    --ntasks=8 \
    --ntasks-per-node=8 \
    --cpus-per-task=2 \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-37 \
    python main.py -m dab_detr \
    --batch_size 2 \
    --epochs 50 \
    --lr_drop 40 \
    --random_refpoints_xy \
    --output_dir ${EXP_DIR} \
     2>&1 | tee ${EXP_DIR}/detailed_log.txt
