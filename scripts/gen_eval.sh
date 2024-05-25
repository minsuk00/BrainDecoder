CKPT="/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/2024-05-05_06:45:07/version/checkpoints/epoch=708_val_loss=0.4873.ckpt"

python /home/choi/BrainDecoder/code/gen_eval.py \
    --ckpt ${CKPT} \
    --row -1 \
    --col 1

# use_pooling, eval, uc_scale, seed, ddim_steps

# 15933MiB / 32510MiB cuda:0
# 1001.4379305839539 seconds for 100 rows, 1 col
# 2000 seconds for 100 rows, 2 col
# 2:12:01.342036 seconds for 300 rows, 1 col

# 1 row, 1 col only IS : 0.37624454498291016 eval time: 1
# 10 row, 1 col only IS : 0.3926832675933838 eval time: 5.74
# 100 row, 1 col only IS : 0.4838085174560547  eval time: 0.385
# -1 row 1 col only IS: 6.765 eval time: 20:56

# 20 row 1 col:
# SSIM evaluation finished: 32.705912351608276
# IS evaluation finished: 0.48696422576904297
# Evaluation complete: 44.32

# 10 row 1 col:
# SSIM evaluation finished: 8.082582712173462
# IS evaluation finished: 0.46749114990234375
# Evaluation complete: 0:00:14.171999 elapsed