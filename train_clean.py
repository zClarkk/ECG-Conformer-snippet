import script_gpu
script_gpu.mask_unused_gpus()
import os
import math
import json
import argparse
import time
import numpy
import yaml
import csv
import time
from csv import writer
from tfcad.utils.environment import setup_environment, setup_strategy
from tfcad.losses import CategoricalFocalCrossEntropy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import fileinput
import custom_focal_loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #hides some tf messages

### Setting up parser and configs
DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.yml')
id_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'id_file.txt')
run_id_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'run_id.txt')
tf.keras.backend.clear_session()
parser = argparse.ArgumentParser(prog='DNN Training')
parser.add_argument('--config', type=str, default=DEFAULT_YAML, help='The file path of model configuration file')
parser.add_argument('--devices', type=int, nargs='*', default=[0], help='Devices ids to apply distributed training')
parser.add_argument('--mxp', default=False, action='store_true', help='Enable mixed precision')
args = parser.parse_args()
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': args.mxp})
setup_environment()
strategy = setup_strategy(args.devices)
import tensorflow_addons as tfa
from tfcad.configs.configs import Config
from tfcad.datasets.datasets import ECGDataset
from tfcad.featurizers.featurizers import SignalFeaturizer, ClassFeaturizer
from tfcad.models.conformer import Conformer
from tfcad.utils.utils import get_avg_result
from tfcad.optimizers.schedules import TransformerSchedule
from sklearn.metrics import accuracy_score
config = Config(args.config)
signal_featurizer = SignalFeaturizer(config.signal_config)
class_featurizer = ClassFeaturizer(config.class_config)
dataset = ECGDataset(**vars(config.train_dataset_config), signal_featurizer=signal_featurizer, class_featurizer=class_featurizer)


run_name = input('Name this run:\n')
save_results = True
paralell_run = True
### Inits
epochs = 15
if save_results:
    with open(run_id_file, 'r') as reader:
        run_id = int(reader.read())
    with open(id_file, 'r') as reader:
        id = int(reader.read())
if save_results and paralell_run:
    run_id_p = run_id + 1
    id_p = id + 1
    with open(run_id_file, 'w') as writer:
        writer.write(str(run_id_p))
    with open(id_file, 'w') as writer:
        writer.write(str(id_p))
#with strategy.scope():
global_batch_size = 16 #was 512
global_batch_size *= strategy.num_replicas_in_sync
for i, fold in enumerate(dataset.create(global_batch_size)):
    ### Inits
    labels = []
    predictions = []
    AF, AFIB, SA, SB, SR, ST, SVT = 0, 0, 0, 0, 0, 0, 0
    AF_P, AFIB_P, SA_P, SB_P, SR_P, ST_P, SVT_P = 0, 0, 0, 0, 0, 0, 0

    ### Assigning and initializing datasets & model
    train_dataset, test_dataset = fold
    conformer = Conformer(**config.model_config)
    conformer._build(signal_featurizer.shape, batch_size=global_batch_size)
    conformer.summary(line_length=100)

    ### Optimizer settings
    optimizer = tf.keras.optimizers.Adam(
        TransformerSchedule(
            d_model=conformer.dmodel,
            warmup_steps=config.optimizer_config["warmup_steps"],
            max_lr=tf.cast((0.0001 / math.sqrt(conformer.dmodel)), tf.float32) #was 0.05 or 0.0001 SHOULD TEST 0.001
        ),
        beta_1=config.optimizer_config["beta1"],
        beta_2=config.optimizer_config["beta2"],
        epsilon=config.optimizer_config["epsilon"]
    )

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = conformer(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, conformer.trainable_weights)
        optimizer.apply_gradients(zip(grads, conformer.trainable_weights))
        train_acc_metric.update_state(y, logits)
        return loss_value
    #@tf.function
    def test_step(x, y):
        val_logits = conformer(x, training=False)
        labels_predicted_batch = tf.argmax(val_logits, axis=1).numpy()
        predictions.extend(labels_predicted_batch)
        val_acc_metric.update_state(y, val_logits)

    ### Assigning error metrics & loss function
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric = keras.metrics.CategoricalAccuracy()
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.AUTO) #CE
    #loss_fn = CategoricalFocalCrossEntropy() # file focal loss CF
    #loss_fn = custom_focal_loss.focal_loss() # file custom focal loss FL
    #loss_fn = tf.keras.losses.BinaryCrossentropy()

    ### Training call
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train)
            if step % 2 == 0:
                #print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))
                print("Seen so far: %s samples" % ((step + 1) * global_batch_size))
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_acc_metric.reset_states()

    ### Test call
    for x_batch_val, y_batch_val in test_dataset:
        labels_batch = tf.argmax(y_batch_val, axis=1).numpy()
        labels.extend(labels_batch)
        test_step(x_batch_val, y_batch_val)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    save_val_acc = float(val_acc)
    print("Time taken: %.2fs" % (time.time() - start_time))

    ### Calculating accuracy again for comparison
    my_accuracy = accuracy_score(labels, predictions)

    print("NO LOOK HERE:", my_accuracy)
    print("Finished fold:", i)

    ### Counting labels
    for label in labels:
        if label == 0: AF += 1
        if label == 1: AFIB += 1
        if label == 2: SA += 1
        if label == 3: SB += 1
        if label == 4: SR += 1
        if label == 5: ST += 1
        if label == 6: SVT += 1

    ### Counting predictions
    for pred in predictions:
        if pred == 0: AF_P += 1
        if pred == 1: AFIB_P += 1
        if pred == 2: SA_P += 1
        if pred == 3: SB_P += 1
        if pred == 4: SR_P += 1
        if pred == 5: ST_P += 1
        if pred == 6: SVT_P += 1

    ### Configuring dict for stats json and time
    fold_stats = {"epoch": epochs, "my_accuracy": my_accuracy, "val_acc": save_val_acc,
                  "AF": AF, "AFIB": AFIB, "SA": SA, "SB": SB, "SR": SR, "ST": ST, "SVT": SVT,
                  "AF_P": AF_P, "AFIB_P": AFIB_P, "SA_P": SA_P, "SB_P": SB_P, "SR_P": SR_P, "ST_P": ST_P, "SVT_P": SVT_P}
    timestr = time.strftime("%d%m%Y-%H%M%S")

    ### Printing labels, predictions & fold stats
    if save_results:
        with open('/home/erguen/tf-arrhythmia-detection-main/examples/conformer/result/labels' + '_ID' + str(run_id) + run_name + '.csv','a+', newline='') as g:
            writer = csv.writer(g)
            writer.writerows(map(lambda xx: [xx], labels))
        with open('/home/erguen/tf-arrhythmia-detection-main/examples/conformer/result/predictions' + '_ID' + str(run_id) + run_name + '.csv', 'a+',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(map(lambda xx: [xx], predictions))
        with tf.io.gfile.GFile(os.path.join(config.learning_config.result_dir, "ID" + str(id) + "_" + timestr + run_name + '.json'),'w') as fout:
            json.dump(fold_stats, fout)

    # Clearing session to avoid core dump
    tf.keras.backend.clear_session()
    # break

### Increase ids by one and write them into the respective file
if save_results and not paralell_run:
    run_id += 1
    id += 1
    with open(run_id_file, 'w') as writer:
        writer.write(str(run_id))
    with open(id_file, 'w') as writer:
        writer.write(str(id))