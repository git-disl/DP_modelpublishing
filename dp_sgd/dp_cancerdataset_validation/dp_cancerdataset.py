# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example differentially private trainer and evaluator for MNIST.
"""
from __future__ import division

import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'


import numpy as np
import tensorflow as tf
import adpallocater as adp
from datetime import datetime
from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.privacy_accountant.tf import accountant
from differential_privacy.datasets.cancer import print_csv_tfrecords

NUM_TRAINING_IMAGES = 560
NUM_TESTING_IMAGES = 63
NUM_VALIDATION_IMAGES = 60 # totol 150
INPUT_SIZE = 9
LABEL_SIZE = 2

# parameters for the training
tf.flags.DEFINE_integer("batch_size", NUM_TRAINING_IMAGES,
                        "The training batch size.")
tf.flags.DEFINE_integer("batches_per_lot", 1,
                        "Number of batches per lot.")
# Together, batch_size and batches_per_lot determine lot_size.
tf.flags.DEFINE_integer("num_training_steps", 1000,
                        "The number of training steps."
                        "This counts number of lots.")

tf.flags.DEFINE_bool("randomize", False,
                     "If true, randomize the input data; otherwise use a fixed "
                     "seed and non-randomized input.")
tf.flags.DEFINE_bool("freeze_bottom_layers", False,
                     "If true, only train on the logit layer.")
tf.flags.DEFINE_bool("save_mistakes", False,
                     "If true, save the mistakes made during testing.")
tf.flags.DEFINE_float("lr", 0.01, "start learning rate")
tf.flags.DEFINE_float("end_lr", 0.01, "end learning rate")
tf.flags.DEFINE_float("lr_saturate_epochs", 0,
                      "learning rate saturate epochs; set to 0 for a constant "
                      "learning rate of --lr.")


tf.flags.DEFINE_float("default_gradient_l2norm_bound", 4.0, "norm clipping")

tf.flags.DEFINE_string("training_data_path",
                       "./differential_privacy/datasets/cancer_validation/cancer_train.csv.tfrecords",
                       "Location of the training data.")
tf.flags.DEFINE_string("eval_data_path",
                       "./differential_privacy/datasets/cancer_validation/cancer_test_shuffle.csv.tfrecords",
                       "Location of the eval data.")
tf.flags.DEFINE_string("validation_data_path",
                        "./differential_privacy/datasets/cancer_validation/cancer_valid_shuffle.csv.tfrecords",
                        "Location of the validation data.")
tf.flags.DEFINE_integer("eval_steps", 20,
                        "Evaluate the model every eval_steps")

# Parameters for privacy spending. We allow linearly varying eps during
# training.
tf.flags.DEFINE_string("accountant_type", "Moments", "Moments, Amortized.")

# Flags that control privacy spending during training.
tf.flags.DEFINE_float("eps", 1.0,
                      "Start privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("end_eps", 1.0,
                      "End privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("eps_saturate_epochs", 0,
                      "Stop varying epsilon after eps_saturate_epochs. Set to "
                      "0 for constant eps of --eps. "
                      "Used if accountant_type is Amortized.")
tf.flags.DEFINE_float("delta", 1e-5,
                      "Privacy spending for training. Constant through "
                      "training, used if accountant_type is Amortized.")

tf.flags.DEFINE_float("sigma", 8.0,
                      "Noise sigma, used only if accountant_type is Moments")


tf.flags.DEFINE_string("target_eps", "0.125,0.25,0.5,1,2,4,8",
                       "Log the privacy loss for the target epsilon's. Only "
                       "used when accountant_type is Moments.")
tf.flags.DEFINE_float("target_delta", 1e-5,
                      "Maximum delta for --terminate_based_on_privacy.")
tf.flags.DEFINE_bool("terminate_based_on_privacy", True,
                     "Stop training if privacy spent exceeds "
                     "(max(--target_eps), --target_delta), even "
                     "if --num_training_steps have not yet been completed.")

tf.flags.DEFINE_string("save_path", "./save/cancer_dir",
                       "Directory for saving model outputs.")

FLAGS = tf.flags.FLAGS


def DataInput(data_file, batch_size, randomize):
  """Create operations to read the input file.

  Args:
   data_file: Path of a file containing the Iris data.
    batch_size: size of the mini batches to generate.
    randomize: If true, randomize the dataset.

  Returns:
    features: A tensor with the formatted image data. shape [batch_size, 4]
    labels: A tensor with the labels for each image.  shape [batch_size]
  """
  file_queue = tf.train.string_input_producer([data_file])
  reader = tf.TFRecordReader()
  _, value = reader.read(file_queue)
  example = tf.parse_single_example(
      value,
      features={"features": tf.FixedLenFeature(shape=[INPUT_SIZE], dtype=tf.float32),
                "label": tf.FixedLenFeature([1], tf.float32)})
  features = example["features"]
  labels = tf.cast(example["label"], dtype=tf.int32)
  labels = tf.reshape(labels, [])

  if randomize:
      features, labels = tf.train.shuffle_batch(
        [features, labels], batch_size=batch_size,
        capacity=(batch_size * 100),
        min_after_dequeue=(batch_size * 10))
  else:
      features, labels = tf.train.batch([features, labels], batch_size=batch_size)

  return features, labels


def Eval(data_file, network_parameters, num_testing_images,
         randomize, load_path, save_mistakes=False):
  """Evaluate MNIST for a number of steps.

  Args:
    mnist_data_file: Path of a file containing the MNIST images to process.
    network_parameters: parameters for defining and training the network.
    num_testing_images: the number of images we will evaluate on.
    randomize: if false, randomize; otherwise, read the testing images
      sequentially.
    load_path: path where to load trained parameters from.
    save_mistakes: save the mistakes if True.

  Returns:
    The evaluation accuracy as a float.
  """
  # Like for training, we need a session for executing the TensorFlow graph.

  batch_size = NUM_TESTING_IMAGES

  with tf.Graph().as_default(), tf.Session() as sess:
    # Create the basic Mnist model.
    features, labels = DataInput(data_file, batch_size, False)
    logits, _, _ = utils.BuildNetwork(features, network_parameters)
    softmax = tf.nn.softmax(logits)

    # Load the variables.
    ckpt_state = tf.train.get_checkpoint_state(load_path)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      raise ValueError("No model checkpoint to eval at %s\n" % load_path)

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    coord = tf.train.Coordinator()
    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    total_examples = 0
    correct_predictions = 0
    image_index = 0
    mistakes = []
    for _ in range((num_testing_images + batch_size - 1) // batch_size):
      predictions, label_values = sess.run([softmax, labels])

      # Count how many were predicted correctly.
      for prediction, label_value in zip(predictions, label_values):
        total_examples += 1
        if np.argmax(prediction) == label_value:
          correct_predictions += 1
        elif save_mistakes:
          mistakes.append({"index": image_index,
                           "label": label_value,
                           "pred": np.argmax(prediction)})
        image_index += 1

  return (correct_predictions / total_examples,
          mistakes if save_mistakes else None)


def Train(train_file, test_file, validation_file, network_parameters, num_steps,
          save_path, total_rho, eval_steps=0):
  """Train MNIST for a number of steps.

  Args:
    mnist_train_file: path of MNIST train data file.
    mnist_test_file: path of MNIST test data file.
    network_parameters: parameters for defining and training the network.
    num_steps: number of steps to run. Here steps = lots
    save_path: path where to save trained parameters.
    eval_steps: evaluate the model every eval_steps.

  Returns:
    the result after the final training step.

  Raises:
    ValueError: if the accountant_type is not supported.
  """
  batch_size = NUM_TRAINING_IMAGES

  params = {"accountant_type": FLAGS.accountant_type,
            "task_id": 0,
            "batch_size": FLAGS.batch_size,
            "default_gradient_l2norm_bound":
            network_parameters.default_gradient_l2norm_bound,
            "num_examples": NUM_TRAINING_IMAGES,
            "learning_rate": FLAGS.lr,
            "end_learning_rate": FLAGS.end_lr,
            "learning_rate_saturate_epochs": FLAGS.lr_saturate_epochs
           }
  # Log different privacy parameters dependent on the accountant type.
  if FLAGS.accountant_type == "Amortized":
    params.update({"flag_eps": FLAGS.eps,
                   "flag_delta": FLAGS.delta,
                  })
  elif FLAGS.accountant_type == "Moments":
    params.update({"sigma": FLAGS.sigma,
                  })

  with tf.device('/gpu:0'), tf.Graph().as_default(), tf.Session() as sess:
    #print_csv_tfrecords.print_tfrecords(train_file)
    features, labels = DataInput(train_file, batch_size, False)
    print("network_parameters.input_size", network_parameters.input_size)
    logits, projection, training_params = utils.BuildNetwork(features, network_parameters)

    cost = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.one_hot(labels, LABEL_SIZE))

    # The actual cost is the average across the examples.
    cost = tf.reduce_sum(cost, [0]) / batch_size

    if FLAGS.accountant_type == "Amortized":
      priv_accountant = accountant.AmortizedAccountant(NUM_TRAINING_IMAGES)
      sigma = None
    elif FLAGS.accountant_type == "Moments":
      priv_accountant = accountant.GaussianMomentsAccountant(
          NUM_TRAINING_IMAGES)
      sigma = FLAGS.sigma
    elif FLAGS.accountant_type == "ZDCP":
        priv_accountant = accountant.zCDPAccountant
    else:
      raise ValueError("Undefined accountant type, needs to be "
                       "Amortized or Moments, but got %s" % FLAGS.accountant)
    # Note: Here and below, we scale down the l2norm_bound by
    # batch_size. This is because per_example_gradients computes the
    # gradient of the minibatch loss with respect to each individual
    # example, and the minibatch loss (for our model) is the *average*
    # loss over examples in the minibatch. Hence, the scale of the
    # per-example gradients goes like 1 / batch_size.
    gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(
        priv_accountant,
        [network_parameters.default_gradient_l2norm_bound / batch_size, True])

    for var in training_params:
      if "gradient_l2norm_bound" in training_params[var]:
        l2bound = training_params[var]["gradient_l2norm_bound"] / batch_size
        gaussian_sanitizer.set_option(var,
                                      sanitizer.ClipOption(l2bound, True))
    lr = tf.placeholder(tf.float32)
    eps = tf.placeholder(tf.float32)
    delta = tf.placeholder(tf.float32)
    varsigma = tf.placeholder(tf.float32, shape=[])

    init_ops = []

    # Add global_step
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                              name="global_step")
    with_privacy = True

    if with_privacy:
        gd_op = dp_optimizer.DPGradientDescentOptimizer(
          lr,
          [eps, delta],
          gaussian_sanitizer,
          varsigma,
          batches_per_lot=FLAGS.batches_per_lot).minimize(
              cost, global_step=global_step)
    else:
        print("No privacy")
        gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    # We need to maintain the intialization sequence.
    for v in tf.trainable_variables():
      sess.run(tf.variables_initializer([v]))
    sess.run(tf.global_variables_initializer())
    sess.run(init_ops)

    results = []
    start_time = time.time()
    prev_time = start_time
    filename = "results-0.json"
    log_path = os.path.join(save_path, filename)

    target_eps = [float(s) for s in FLAGS.target_eps.split(",")]
    if FLAGS.accountant_type == "Amortized":
      # Only matters if --terminate_based_on_privacy is true.
      target_eps = [max(target_eps)]
    max_target_eps = max(target_eps)

    lot_size = FLAGS.batches_per_lot * FLAGS.batch_size
    lots_per_epoch = NUM_TRAINING_IMAGES / lot_size
    previous_epoch=-1
    rho_tracking=[0]

    validation_accuracy_list =[]
    previous_validaccuracy = 0
    tracking_sigma=[]


    curr_sigma=35

    for step in range(num_steps):
      epoch = step // lots_per_epoch
      curr_lr = utils.VaryRate(FLAGS.lr, FLAGS.end_lr,
                               FLAGS.lr_saturate_epochs, epoch)
      curr_eps = utils.VaryRate(FLAGS.eps, FLAGS.end_eps,
                                FLAGS.eps_saturate_epochs, epoch)
      if with_privacy:
        old_sigma = curr_sigma

        #total budget
        rhototal=total_rho

        # validation based decay
        # period=10,  threshold=0.01, decay_factor=0.9
        period = 20
        decay_factor = 0.99
        threshold = 0.01
        m = 1
        if epoch - previous_epoch == 1 and (epoch + 1) % period == 0:  # checking epoch
            current_validaccuracy = sum(validation_accuracy_list[-m:]) / m
            if current_validaccuracy - previous_validaccuracy < threshold:
                curr_sigma = decay_factor * curr_sigma
            previous_validaccuracy = current_validaccuracy

        if old_sigma != curr_sigma:
            print(curr_sigma)

        # for tracking by epoch
        if epoch - previous_epoch == 1:
            tracking_sigma.append(curr_sigma)
            rho_tracking.append(rho_tracking[-1] + 1 / (2.0 * curr_sigma ** 2))
            previous_epoch = epoch
            if with_privacy == True and rho_tracking[-1] > rhototal:
                print("stop at epoch%d" % epoch)
                break
            print(rho_tracking)
            print(tracking_sigma)


        if step%100==0:
            print(curr_sigma)
            print(rho_tracking[-1])


      for _ in range(FLAGS.batches_per_lot):
        _ = sess.run(
            [gd_op], feed_dict={lr: curr_lr, eps: curr_eps, delta: FLAGS.delta, varsigma: curr_sigma})
      sys.stderr.write("step: %d\n" % step)

      # See if we should stop training due to exceeded privacy budget:
      should_terminate = False

      if (eval_steps > 0 and (step + 1) % eval_steps == 0) or should_terminate:
        saver.save(sess, save_path=save_path + "/ckpt")
        train_accuracy, _ = Eval(train_file, network_parameters,
                                 num_testing_images=NUM_TRAINING_IMAGES,
                                 randomize=False, load_path=save_path)
        sys.stderr.write("train_accuracy: %.2f\n" % train_accuracy)
        test_accuracy, mistakes = Eval(test_file, network_parameters,
                                       num_testing_images=NUM_TESTING_IMAGES,
                                       randomize=False, load_path=save_path,
                                       save_mistakes=FLAGS.save_mistakes)
        sys.stderr.write("eval_accuracy: %.2f\n" % test_accuracy)
        validation_accuracy, mistakes = Eval(validation_file, network_parameters,
                                       num_testing_images=NUM_VALIDATION_IMAGES,
                                       randomize=False, load_path=save_path,
                                       save_mistakes=FLAGS.save_mistakes)
        sys.stderr.write("validation_accuracy: %.2f\n" % validation_accuracy)
        validation_accuracy_list.append(validation_accuracy)

        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        prev_time = curr_time

        results.append({"step": step+1,  # Number of lots trained so far.
                        "elapsed_secs": elapsed_time,
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                        "mistakes": mistakes})
        loginfo = {"elapsed_secs": curr_time-start_time,
                   "train_accuracy": train_accuracy,
                   "test_accuracy": test_accuracy,
                   "num_training_steps": step+1,  # Steps so far.
                   "mistakes": mistakes,
                   "result_series": results}
        loginfo.update(params)
        if log_path:
          with tf.gfile.Open(log_path, "w") as f:
            json.dump(loginfo, f, indent=2)
            f.write("\n")
            f.close()

      if should_terminate:
        break


    print(rho_tracking[:-1])
    saver.save(sess, save_path=save_path + "/ckpt")
    train_accuracy, _ = Eval(train_file, network_parameters,
                             num_testing_images=NUM_TRAINING_IMAGES,
                             randomize=False, load_path=save_path)
    sys.stderr.write("train_accuracy: %.2f\n" % train_accuracy)
    test_accuracy, mistakes = Eval(test_file, network_parameters,
                                   num_testing_images=NUM_TESTING_IMAGES,
                                   randomize=False, load_path=save_path,
                                   save_mistakes=FLAGS.save_mistakes)
    sys.stderr.write("eval_accuracy: %.2f\n" % test_accuracy)

    curr_time = time.time()
    elapsed_time = curr_time - prev_time
    prev_time = curr_time

    results.append({"step": step + 1,  # Number of lots trained so far.
                    "elapsed_secs": elapsed_time,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "mistakes": mistakes})
    loginfo = {"elapsed_secs": curr_time - start_time,
               "train_accuracy": train_accuracy,
               "test_accuracy": test_accuracy,
               "num_training_steps": step,  # Steps so far.
               "mistakes": mistakes,
               "result_series": results}
    loginfo.update(params)
    if log_path:
        with tf.gfile.Open(log_path, "w") as f:
            json.dump(loginfo, f, indent=2)
            f.write("\n")
            f.close()

def main(_):
  network_parameters = utils.NetworkParameters()

  # If the ASCII proto isn't specified, then construct a config protobuf based
  # on 3 flags.
  network_parameters.input_size = INPUT_SIZE
  network_parameters.projection_type = "NONE"
  network_parameters.default_gradient_l2norm_bound = (
      FLAGS.default_gradient_l2norm_bound)
  hidden_units=[10, 20, 10]
  num_hidden_layers =3
  for i in range(num_hidden_layers):
    hidden = utils.LayerParameters()
    hidden.name = "hidden%d" % i
    hidden.num_units = hidden_units[i]
    hidden.relu = True
    hidden.with_bias = True
    hidden.trainable = True
    network_parameters.layer_parameters.append(hidden)

  logits = utils.LayerParameters()
  logits.name = "logits"
  logits.num_units = LABEL_SIZE
  logits.relu = False
  logits.with_bias = False
  network_parameters.layer_parameters.append(logits)

  Train(FLAGS.training_data_path,
        FLAGS.eval_data_path,
        FLAGS.validation_data_path,
        network_parameters,
        FLAGS.num_training_steps,
        FLAGS.save_path,
        total_rho=1.0/(2.0*25**2)*500.0,
        eval_steps=FLAGS.eval_steps)


if __name__ == "__main__":
  tf.app.run()
