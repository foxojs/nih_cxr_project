# config.py
IMAGE_SIZE = (256, 256)
PATCH_SIZE =8
HIDDEN_SIZE = 128
NUM_LAYERS = 8
NUM_HEADS = 8
NUM_CLASSES = 15
<<<<<<< HEAD
BATCH_SIZE =512 
LEARNING_RATE = 0.001
NUM_EPOCHS =20 
DS_TRAIN_SIZE = "train[:60000]"
DS_TEST_SIZE = "test[:20000]"
=======
BATCH_SIZE =16 
LEARNING_RATE = 0.001
NUM_EPOCHS =5 
DS_TRAIN_SIZE = "train[:2000]"
DS_TEST_SIZE = "test[:1000]"
>>>>>>> ddcb71fb8ff0995753f42b3a7030f1eecc8354b3
USE_PRETRAINED = False
MODEL_NAME = "vit_b_16"
FINE_TUNE_LAYERS = 10 
RESULTS_PATH = "../vit_base_model/results"
MODEL_SAVE_PATH = "../trained_models/best_model.pth"
TRAINING_LOGS_PATH = "results/training_logs.csv"
CONFUSION_MATRIX_PATH = "results/plots/confusion_matrices"
METRICS_CSV_PATH = "results/df/df_metrics_overall.csv"
