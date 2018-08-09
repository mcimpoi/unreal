# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from environment.environment import Environment
from model.model import UnrealModel
from train.experience import ExperienceFrame
from options import get_options

from time import sleep

# get command line args
flags = get_options("evaluate")

class Evaluate(object):
  def __init__(self):
    self.action_size = Environment.get_action_size(flags.env_type, flags.env_name)
    self.objective_size = Environment.get_objective_size(flags.env_type, flags.env_name)
    print('flags:use_pixel_change {}'.format(flags.use_pixel_change))
    sleep(10)
    self.global_network = UnrealModel(self.action_size,
                                      self.objective_size,
                                      -1,
                                      flags.use_lstm,
                                      flags.use_pixel_change,
                                      flags.use_value_replay,
                                      flags.use_reward_prediction,
                                      0.0,
                                      0.0,
                                      "/cpu:0",
                                      for_display=True)
    self.environment = Environment.create_environment(flags.env_type, flags.env_name,
                                                      env_args={'episode_schedule': flags.split,
                                                                'log_action_trace': flags.log_action_trace,
                                                                'max_states_per_scene': flags.episodes_per_scene,
                                                                'episodes_per_scene_test': flags.episodes_per_scene})
    print('\n======\nENV in Evaluate::ctor')
    print(self.environment)
    print(self.global_network)
    print('val_replay!!! {}'.format(flags.use_value_replay))
    print(flags.split)
    print('=======\n')
    sleep(10)
    self.episode_reward = 0

  def update(self, sess):
    self.process(sess)

  def is_done(self):
    return self.environment.is_all_scheduled_episodes_done()

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def process(self, sess):
    last_action = self.environment.last_action
    last_reward = np.clip(self.environment.last_reward, -1, 1)
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action, self.action_size,
                                                                  last_reward, self.environment.last_state)
    
    if not flags.use_pixel_change:
      pi_values, v_value = self.global_network.run_base_policy_and_value(sess,
                                                                         self.environment.last_state,
                                                                         last_action_reward)
    else:
      pi_values, v_value, pc_q = self.global_network.run_base_policy_value_pc_q(sess,
                                                                                self.environment.last_state,
                                                                                last_action_reward)
    action = self.choose_action(pi_values)
    state, reward, terminal, pixel_change = self.environment.process(action)
    self.episode_reward += reward
  
    if terminal:
      self.environment.reset()
      self.episode_reward = 0


def main(args):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01, allow_growth=False)  # avoid using all gpu memory
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess.run(tf.global_variables_initializer())
  
  evaluate = Evaluate()
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)

  print('\nFlags: pixel_change: {} Split: {}'.format(
    flags.use_pixel_change, flags.split))
  print('\n\n----\n', checkpoint)
  sleep(5)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)


    from tensorflow.python import pywrap_tensorflow
    checkpoint_path = checkpoint.model_checkpoint_path
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in var_to_shape_map:
      print("tensor_name: ", key)
#      print(reader.get_tensor(key))
              
    print('\n==============\n')
    sleep(10)
    
    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    print(checkpoint)
    sleep(5)
  else:
    print("Could not find old checkpoint")

  while not evaluate.is_done():
    evaluate.update(sess)

    
if __name__ == '__main__':
  tf.app.run()
