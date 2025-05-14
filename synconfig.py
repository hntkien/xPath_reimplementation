import os 
import logging 

# ========== Configurations ========== #
HGN_TYPE = 'simplehgn'
DATASET = 'syn_imdb'
TARGET_NTYPE = '0' # syntehtic datasets define the target node type as '0'
# N_LAYER = 3
REPEAT_ID = 4 # Experiment id
GPU = 0

# Default configuration from SimpleHGN Paper
if DATASET == 'syn_dblp' or DATASET == 'syn_acm':
    N_LAYER = 3
    s = 0.05
elif DATASET == 'syn_imdb':
    N_LAYER = 3
    s = 0.1

# Define Paths 
abs_path = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(abs_path, 'log')
os.makedirs(log_dir, exist_ok=True)
data_dir = os.path.join(abs_path, 'data', DATASET)
result_dir = os.path.join(abs_path, 'results', HGN_TYPE)
os.makedirs(result_dir, exist_ok=True)
ckpt_dir = os.path.join(abs_path, 'ckpt', DATASET)
os.makedirs(ckpt_dir, exist_ok=True)

# Define xPath Hyperparameters
if DATASET == 'syn_acm':
    XPATH_BEAM = 5
    XPATH_SAMPLE_N = 5
    XPATH_TOP_K = 4
else:
    XPATH_BEAM = 2
    XPATH_SAMPLE_N = 10
    XPATH_TOP_K = 4

pred_list_path = f"{ckpt_dir}/{HGN_TYPE}_{DATASET}_pred_list_{N_LAYER}.json"

# To use trained model:
bk_dir = f"{ckpt_dir}/bk"
os.makedirs(bk_dir, exist_ok=True)
hgn_path = f"{bk_dir}/{HGN_TYPE}_{DATASET}_{N_LAYER}"
graph_path = f"{data_dir}/{DATASET}_processed_graph.bin"
result_path = f'{result_dir}/{DATASET}_l{N_LAYER}_xpath2s_{XPATH_BEAM}_{XPATH_SAMPLE_N}_exp{REPEAT_ID}'
cause_node_dict_path = os.path.join(ckpt_dir, f'cause_node_dict_{DATASET}_l{N_LAYER}_exp{REPEAT_ID}.json')

# Init logger
log_root = log_dir + f'/{HGN_TYPE}'
os.makedirs(log_root, exist_ok=True)
log_file = log_root + f'/{DATASET}_l{N_LAYER}_r{REPEAT_ID}.log'
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()
fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger = logging.getLogger('updateSecurity')
logger.setLevel('DEBUG')
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ========== End of Configurations ========== #

# ========== Configuration Dictionaries ========== #
PATHS = {
    'graph_path': graph_path,
    'hgn_path': hgn_path,
    'result_path': result_path,
    'pred_list_path': pred_list_path,
    'cause_node_dict_path': cause_node_dict_path
}

DATASET_CONFIG = {
    "dataset_name": DATASET,
    "target_ntype": TARGET_NTYPE,
    "force_reload": True,
}

MODEL_CONFIG = {
    "hgn_type": HGN_TYPE,
    "n_layer": N_LAYER,
    "gpu": GPU,
    "neg_slope": s,
}

EXPLAIN_CONFIG = {
    "xpath_beam": XPATH_BEAM,
    "xpath_sample_n": XPATH_SAMPLE_N,
    "xpath_top_k": XPATH_TOP_K,
}