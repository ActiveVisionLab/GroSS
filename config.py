from yacs.config import CfgNode as CN


_C = CN()
_C.OUT_DIR = "./runs"
_C.NEW_DIR = "./new_runs"

# SYSTEM
_C.SYSTEM = CN()
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 8
_C.SYSTEM.GPU_NUMBER = 0


# DATASET
_C.DATASET = CN()
_C.DATASET.NAME = "cifar10"
_C.DATASET.ROOT_DIR = "./data/cifar10"
_C.DATASET.NORM_MEAN = [0.5, 0.5, 0.5]
_C.DATASET.NORM_VAR = [0.5, 0.5, 0.5]
_C.DATASET.INPUT_SIZE = (3, 32, 32)


# MODEL
_C.MODEL = CN()
_C.MODEL.BACKBONE = "4Conv"
_C.MODEL.LOAD_PATH = ""
_C.MODEL.RUN_SPECIFIC_LOAD_PATH = False
_C.MODEL.INIT_DECOMPOSED = False
_C.MODEL.INIT_FUSED = False
_C.MODEL.FUSE_FOR_TRAIN = False
_C.MODEL.DECOMPOSABLE_LAYERS = ["conv2", "conv3", "conv4"]
# Group sizes each layer will be decomposed into
# empty for powers of 2 <= in_channels
_C.MODEL.DECOMPOSE_STRUCTURE = [[], [], []]
_C.MODEL.BOTTLE_DIMS = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,]


# DECOMPOSITION
_C.DECOMPOSITION = CN()
# Stopping criteria
_C.DECOMPOSITION.MIN_STEP = 1e-6
_C.DECOMPOSITION.MAX_ITER = 50000
# Number of iterations that are included in the step rolling average
_C.DECOMPOSITION.STEP_AVG_LENGTH = 5


# TRAINING
_C.TRAIN = CN()
# General training parameters
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.NUM_EPOCHS = 80
_C.TRAIN.INITIAL_LR = 0.1
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.FIXED_EPOCH_SIZE = False
_C.TRAIN.EPOCH_SIZE = 10000

# Epoch number at which learning rate is decreased
_C.TRAIN.LR_MILESTONES = [40, 60]
_C.TRAIN.LR_GAMMA = 0.1
# Number of epochs between saving models
_C.TRAIN.SAVE_AT_START = False
_C.TRAIN.SAVE_EVERY = 20
_C.TRAIN.EARLY_STOPPING = False
# Number of times to repeat the training procedure
_C.TRAIN.NUM_RUNS = 5

# Forward group sizes per layer
# -1 for random sample
_C.TRAIN.GROUP_SIZES = [-1, -1, -1]
# Label names for the summary writer
_C.TRAIN.LOSS_LABEL = "train/loss"
_C.TRAIN.ACC_LABEL = "test/accuracy"
# Additional test config
_C.TRAIN.ADDITIONAL_TEST = False
_C.TRAIN.ADDITIONAL_ACC_LABEL = ""
_C.TRAIN.ADDITIONAL_TEST_GROUP_SIZES = [1, 1, 1]
_C.TRAIN.TEST_EVERY = 1

# Freeze non-decomposed layers
_C.TRAIN.FREEZE_STANDARD_LAYERS = False
_C.TRAIN.FREEZE_BN_LAYERS = False

# Resume from checkpoint
_C.TRAIN.RESUME_PATH = ""


# TESTING
_C.TEST = CN()
# Forward group sizes per layer
# -1 for random sample
_C.TEST.GROUP_SIZES = [32, 32, 64]
_C.TEST.BATCH_SIZE = 1024


# ARCHITECTURE SEARCH
_C.SEARCH = CN()

_C.SEARCH.LOAD_PATH = "{}/{}/decompnet_{}.pth"

_C.SEARCH.POSSIBLE_CONFIGS = [[]]
_C.SEARCH.MAC_LIMIT = -1
_C.SEARCH.BASE_CONFIG = []
_C.SEARCH.LOWER_BOUND = []
_C.SEARCH.LOAD_PRECOMUPTED_RESULTS = False

_C.SEARCH.TYPE = "exhaustive"
_C.SEARCH.POPULATION_SIZE = 10
_C.SEARCH.SAMPLE_SIZE = 5
_C.SEARCH.NUM_ITERS = 10000
_C.SEARCH.OUT_FILE = "search.bin"
_C.SEARCH.NUM_RUNS = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
