# config.py
IMAGE_SIZE = (128, 128)
PATCH_SIZE = 4
HIDDEN_SIZE = 128
NUM_LAYERS = 8
NUM_HEADS = 8
NUM_CLASSES = 15
BATCH_SIZE =64 
LEARNING_RATE = 0.01
NUM_EPOCHS =2 
DS_TRAIN_SIZE = "train[:3000]"
RESULTS_PATH = "../vit_base_model/results"
MODEL_SAVE_PATH = "../trained_models/best_model.pth"
TRAINING_LOGS_PATH = "results/training_logs.csv"
CONFUSION_MATRIX_PATH = "results/plots/confusion_matrices"
METRICS_CSV_PATH = "results/df/df_metrics_overall.csv"
