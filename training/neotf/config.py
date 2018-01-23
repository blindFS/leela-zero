class leela_conf:
    GPU_NUM = 2
    LR = 0.05
    RESIDUAL_FILTERS = 128
    RESIDUAL_BLOCKS = 6
    BATCH_SIZE = 256
    POLICY_LOSS_WEIGHT = 1.0
    MSE_LOSS_WEIGHT = 1.0
    REGULARIZER_SCALE = 0.0001
    INFO_STEP_INTERVAL = 500        # 1000
    EVAL_STEP_INTERVAL = 2000       # 8000
    DATACHANGE_STEP_INTERVAL = 2000
    VALIDATION_STEP_INTERVAL = 2000
    DATA_SIZE = 5000
    DATA_DIR = "./data"
    SAVE_DIR = "./save"
    SAVE_PREFIX = "leelaz-model"
    VALIDATION_LOG = SAVE_DIR + "/val_log"
    VALIDATION_COMMAND = "./validation -u 0 -u 1 -g 2 -n %s -n %s -l -p 100 -t 0.3" \
        + VALIDATION_LOG
