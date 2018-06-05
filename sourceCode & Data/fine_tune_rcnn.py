from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function
import tensorflow as tf
import numpy as np

class AlexNet(object):
  def __init__(self, x, keep_prob, num_classes):
    #x is the input images
    #skip_layer are the layers that will be trained again
    #num_classes is the output classes after finetuing
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    # Call the create function to build the computational graph of AlexNet
    self.create()
  def create(self):
    #conv1
    conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
    
    #lrn1 and pool1 
    #lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha = 0.001/9,beta = 0.75, name = 'norm1')
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],padding ='VALID', name='pool1')
    lrn1 = tf.nn.lrn(pool1,bias=1.0,depth_radius=2 ,alpha = 2e-05,beta = 0.75, name = 'norm1')
    #conv2
    conv2 = conv(lrn1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
    
    #lrn2 and pool2
    #lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha = 0.001/9,beta = 0.75, name = 'norm2')
    pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1],padding ='VALID', name='pool2')
    lrn2 = tf.nn.lrn(pool2,bias = 1.0,depth_radius=2 ,alpha = 2e-05,beta = 0.75, name = 'norm2')
    #conv3
    conv3 = conv(lrn2, 3, 3, 384, 1, 1, name = 'conv3')
    #conv4
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')
    
    #conv5
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
    
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1],padding ='VALID', name='pool5')
    #fcIn
    fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])  
    with tf.variable_scope('fc6') as scope:
      #weights = tf.Variable(tf.truncated_normal([256 * 6 * 6,4096],dtype= tf.float32,stddev= 1e-1),name = 'weights')
      weights = tf.get_variable('weights',shape=[256 * 6 * 6,4096],trainable=True)
      #biases = tf.Variable(tf.constant(0.0, shape=[4096],dtype = tf.float32),trainable= True, name = 'biases')
      biases = tf.get_variable('biases', shape = [4096],trainable=True)
      self.fc6 = tf.nn.xw_plus_b(fcIn,weights,biases,name = scope.name)
      fc6 = tf.nn.relu(self.fc6)
    #set the dropout keepPro 0.5
    dropout1 =  tf.nn.dropout(fc6,self.KEEP_PROB)
    #fc2
    with tf.variable_scope('fc7') as scope:
      #weights = tf.Variable(tf.truncated_normal([4096,4096],dtype= tf.float32,stddev= 1e-1),name = 'weights')
      weights = tf.get_variable('weights',shape=[4096,4096],trainable=True)
      #biases = tf.Variable(tf.constant(0.0, shape=[4096],dtype = tf.float32),trainable= True, name = 'biases')
      biases = tf.get_variable('biases', shape = [4096],trainable=True)
      self.fc7 = tf.nn.xw_plus_b(dropout1,weights,biases,name = scope.name)
      fc7 = tf.nn.relu(self.fc7)
    dropout2 =  tf.nn.dropout(fc7,self.KEEP_PROB)
    #fc3
    with tf.variable_scope('fc8') as scope:
      #weights = tf.Variable(tf.truncated_normal([4096,self.NUM_CLASSES],dtype= tf.float32,stddev= 1e-1),name = 'weights')
      weights = tf.get_variable('weights',shape=[4096,self.NUM_CLASSES],trainable=True)
      #biases = tf.Variable(tf.constant(0.0, shape=[self.NUM_CLASSES],dtype = tf.float32),trainable= True, name = 'biases')
      biases = tf.get_variable('biases', shape = [self.NUM_CLASSES],trainable=True)
      self.fc8 = tf.nn.xw_plus_b(dropout2,weights,biases,name = scope.name)
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
  input_channels = int(x.get_shape()[-1])
  #convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1],padding = padding)
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters],trainable= True)
    biases = tf.get_variable('biases', shape = [num_filters],trainable= True)  
    if groups == 1:
      conv = convolve(x, weights)
    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      # Concat the convolved output together again
      conv = tf.concat(axis = 3, values = output_groups)
    # Add biases 
    #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    bias = tf.nn.bias_add(conv,biases)
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)
    return relu

def print_log(worker_num, arg):
  print("{0}: {1}".format(worker_num, arg))


def map_fun(args, ctx):
  from datetime import datetime
  import math
  import numpy
  import tensorflow as tf
  import time

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
  if job_name == "ps":
    time.sleep((worker_num + 1) * 5)

  # Parameters
  IMAGE_PIXELS = 28
  hidden_units = 128
  batch_size = args.batch_size

  # Get TF cluster and server instances
  cluster, server = ctx.start_cluster_server(1, args.rdma)

  def feed_dict(batch):
    # Convert from [(images, labels)] to two numpy arrays of the proper type
    images = []
    labels = []
    for item in batch:
      images.append(item[0])
      labels.append(item[1])
    xs = numpy.array(images)
    xs = xs.astype(numpy.float32)
    ys = numpy.array(labels)
    ys = ys.astype(numpy.uint8)
    return (xs, ys)

  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % task_index,
      cluster=cluster)):
      LEARING_RATE_BASE = 0.9
      LEARNING_RATE_DECAY = 0.9
      TRAINING_STEPS = 30000
      num_classes= 95 #the classes of our data set butterfly
      train_layers = ['pool1','norm1','conv2','pool2','norm2','conv3','conv4','conv5','pool5','fc6','fc7','fc8']

      global_step = tf.train.get_or_create_global_step()

      #Input and output
      x = tf.placeholder(tf.float32, [None, 227, 227, 3])
      y = tf.placeholder(tf.float32, [None, num_classes])
      keep_prob = tf.placeholder(tf.float32)

      #Initialize model
      model = AlexNet(x, keep_prob, num_classes)

      # output of the AlexNet
      score = model.fc8

      var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
      #loss function
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  
      tf.summary.scalar("loss", loss)

      learning_rate = tf.train.exponential_decay(LEARING_RATE_BASE, global_step,100,LEARNING_RATE_DECAY)
      gradients = tf.gradients(loss, var_list)
      gradients = list(zip(gradients, var_list))
 
      # Create optimizer and apply gradient descent to the trainable variables
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      train_op = optimizer.apply_gradients(grads_and_vars=gradients,global_step=global_step)
      # Test trained model
      label = tf.argmax(y, 1, name="label")
      prediction = tf.argmax(score, 1, name="prediction")
      correct_prediction = tf.equal(prediction, label)

      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
      tf.summary.scalar("acc", accuracy)
      summary_op = tf.summary.merge_all()
    logdir = ctx.absolute_path(args.model)
    print("tensorflow model path: {0}".format(logdir))
    hooks = [tf.train.StopAtStepHook(last_step=100000)]

    if job_name == "worker" and task_index == 0:
      summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    # The MonitoredTrainingSession takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs
    with tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=(task_index == 0),
                                             checkpoint_dir=logdir,
                                             hooks=hooks) as mon_sess:
      step = 0
      #get the feed data from the application
      tf_feed = ctx.get_data_feed(args.mode == "train")
      while not mon_sess.should_stop() and not tf_feed.should_stop() and step < args.steps:
        # Run a training step asynchronously
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        # using feed_dict
        batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
        feed = {x: batch_xs, y: batch_ys, keep_prob:0.5}

        if len(batch_xs) > 0:
          if args.mode == "train":
            _, summary, step = mon_sess.run([train_op, summary_op, global_step], feed_dict=feed)
            # print accuracy and save model checkpoint to HDFS every 100 steps
            if (step % 100 == 0):
              print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step, mon_sess.run(accuracy, {x: batch_xs, y: batch_ys,keep_prob:1.0})))

            if task_index == 0:
              summary_writer.add_summary(summary, step)
          else:  # args.mode == "inference"
            feed = {x: batch_xs, y: batch_ys, keep_prob:1}
            labels, preds, acc = mon_sess.run([label, prediction, accuracy], feed_dict=feed)

            results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l, p in zip(labels, preds)]
            tf_feed.batch_results(results)
            print("results: {0}, acc: {1}".format(results, acc))

      if mon_sess.should_stop() or step >= args.steps:
        tf_feed.terminate()

    # Ask for all the services to stop.
    print("{0} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

  if job_name == "worker" and task_index == 0:
    summary_writer.close()