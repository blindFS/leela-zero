class leela_conf:
    GPU_NUM = 2
    LR = 0.001
    RESIDUAL_FILTERS = 128
    RESIDUAL_BLOCKS = 6
    BATCH_SIZE = 256
    POLICY_LOSS_WEIGHT = 1.0
    MSE_LOSS_WEIGHT = 1.0
    REGULARIZER_SCALE = 0.0001
    # Do actions at certain global step
    INFO_STEP_INTERVAL = 1000
    EVAL_STEP_INTERVAL = 8000
    DATACHANGE_STEP_INTERVAL = 8000
    VALIDATION_STEP_INTERVAL = 24000
    # Max amount of latest self-play matches used for training at once
    DATA_SIZE = 500
    DATA_DIR = "./data"
    SAVE_DIR = "./save"
    SAVE_PREFIX = "leelaz-model"
    VALIDATION_LOG = SAVE_DIR + "/val_log"
    VALIDATION_COMMAND = "./validation -u 0 -u 1 -g 2 -n %s -n %s -p 100 -t 0.2 -l " \
        + VALIDATION_LOG        # ./validation --help for more info
