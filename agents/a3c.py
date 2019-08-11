from jiminy.utils import create_directory
import time
import queue
import threading
import logging
import tensorflow as tf
import numpy as np
# import utils
import sys
# import logging

logging.basicConfig(level=logging.DEBUG,
        filename="app.log",
        mode="w", format="%(name)s - %(levelname)s - %(message)s")

class A3C(object):
    def __init__(self,
            env,
            gamma=0.99,
            momentum=0.999,
            learning_rate=1e-3,
            entropy_beta=1e-3,
            clip_grad=40.,
            logdir='./logs/a2c',
            batch_size=1):
        create_directory(logdir)
        self.logdir = logdir
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_beta = entropy_beta
        self.clip_grad = clip_grad
        self.momentum = momentum
        self.queue_lock = threading.Lock()
        # self.weights_lock = threading.Lock()
        self.grad_queue = queue.Queue()
        self.batch_size = batch_size
        logging.debug("A3C object created")

        self.domnet = env

        self.action_input = tf.keras.layers.Input(shape=[3], dtype=tf.int32,name='action_input')
        self.reward_input = tf.keras.layers.Input(shape=[1], dtype=tf.int32, name='reward_input')

    def create_model(self):
        with self.domnet.graph.as_default():
            bootstrap_input = tf.keras.Input(shape=(1,))
            tf.summary.scalar('episode_reward', bootstrap_input[-1][0])
            model_input = [tf.keras.Input(shape=obj.shape.as_list()[1:], dtype=obj.dtype) for obj in self.domnet.model.inputs]
            action_index = tf.keras.Input(shape=(self.domnet.env.action_space.n,))

            model_output = self.domnet.model(model_input)

            print(model_output[1][0].shape)
            # policy_loss = tf.keras.losses.categorical_crossentropy(bootstrap_input * action_index, model_output[1][0])

            policy_loss = -tf.reduce_sum(tf.reduce_sum(action_index*tf.math.log(model_output[1][0]), axis=-1) * (bootstrap_input - model_output[0]))
            value_loss = tf.keras.losses.MSE(model_output[0], bootstrap_input)
            entropy_loss = tf.keras.losses.categorical_crossentropy(model_output[1][0], model_output[1][0])

            tf.summary.scalar('value_loss', tf.reduce_mean(value_loss))
            tf.summary.scalar('entropy_loss', tf.reduce_mean(entropy_loss))
            loss = 0.5*value_loss + policy_loss # - self.entropy_beta*entropy_loss
            tf.summary.scalar('policy_loss', tf.reduce_mean(policy_loss))
            self.model = tf.keras.Model(inputs=[bootstrap_input] +  model_input + [action_index], outputs=policy_loss)
            def loss_fn(y_true, y_pred):
                return y_pred
            self.model.compile(loss=loss_fn,
                    optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate,
                        clipnorm=1.0))

    @classmethod
    def from_config(cls, config):
        return cls(gamma=config["gamma"], momentum=config["momentum"],
            learning_rate=config["learning_rate"], entropy_beta=config["entropy_beta"],
            clip_grad=config["clip_grad"], logdir=config["logdir"])

    def learn(self):
        logging.debug("Started A3C learner")

        self.writer = tf.summary.FileWriter(self.logdir + "/" + str(time.time()))

        self.domnet.load(self.logdir + "/a3c.h5")
        self.env = self.domnet.env
        self.n_workers = self.env.n
        self.T = 0
        self.episode_count = 0
        self.global_step = tf.train.get_or_create_global_step()
        self.kill_flags = [False for _ in range(self.n_workers)]
        self.thread_list = []
        for i in range(self.n_workers):
            logging.debug("Starting worker {} implementing A2C".format(i))
            t = threading.Thread(target=self.runner_wrapper, args=(i,))
            t.start()
            self.thread_list.append(t)
        self.update_thread = threading.Thread(target=self.update_weights, args=())
        # self.update_thread.start()
        other_fn = {
                self.save : 100,
                self.discount_t(0.99) : 50
                }
        for fn, step in enumerate(other_fn):
            threading.Thread(target=self.loop_step, args=(fn, step)).start()

    def save(self):
        self.domnet.save(self.logdir + "/a3c.h5")

    def loop_step(self, step, fn):
        next_T = 0
        while True:
            if self.T > next_T:
                next_T += step
                fn()
            time.sleep(10)

    def discount_t(self, discount_factor=0.99):
        def fn():
            self.domnet.greedy_epsilon *= discount_factor

        return fn

    def update_weights(self):
            try:
                while True:
                    grad_updates = self.grad_queue.get()
                    grad_updates = zip(grad_updates, self.domnet.model.trainable_weights)
                    with self.weights_lock:
                        self.opt.apply_gradients(grad_updates)
            except:
                self.close()

    def close(self):
        main_thread = threading.currentThread()
        with self.queue_lock:
            self.kill_flags = [True for _ in range(self.n_workers)]
        for t in self.thread_list:
            if t is main_thread :
                continue
            t.join()
        self.update_thread.join()
        self.domnet.save(self.logdir)
        self.env.close()
        logging.debug("Closing A3C master")

    def runner_wrapper(self, *args, **kwargs):
            with self.domnet.graph.as_default():
                with self.domnet.sess.as_default():
                    self.runner(*args, **kwargs)

    def runner(self, index, episode_max_length=100, max_step_count=1e6):
        t,t_s = 0,0
        episode = 0
        action_reset = self.env.action_space.sample()
        while True:
            action_log, state_log, bootstrap_log = dict(), dict(), dict()
            for batch in range(self.batch_size):
                episode += 1
                # noise before running a new episode
                reward_log = dict()
                value_log = dict()
                state_log = dict()
                action_log_prob_log = dict()
                action_log = dict()
                with self.queue_lock:
                    self.global_step.assign_add(1)
                    # exit if the process has run for maximum number of episodes
                    if self.T >= max_step_count: return
                    t_s = t
                obs = self.env.reset_runner(index)
                # action = self.env.action_space.sample()
                while True:
                    time.sleep(0.05)
                    obs, _, _, _ = self.env.step_runner(index, action_reset)
                    if obs is None:
                        continue
                    if obs.query == "":
                        continue
                    break
                print("Starting episode: {}".format(episode))
                self.env.step_runner(index, action_reset)
                done,value = False, None

                # synchoronize model from master
                # model = self.domnet.get_runner_instance()
                flag = True

                # run episode for max_length time steps
                while (not done) and t - t_s < episode_max_length:
                    if obs is None or obs.query == "":
                        obs, _, done, info = self.env.step_runner(index, action_reset)
                        time.sleep(0.02)
                        continue
                    state, value, action, action_log_prob = self.domnet.step_runner(index, obs)
                    if value is None :
                        obs, _, done, info = self.env.step_runner(index, action_reset)
                        time.sleep(0.02)
                        continue
                    value_log[t], action_log_prob_log[t], state_log[t] = value, action_log_prob, state
                    action_log[t] = tf.keras.utils.to_categorical(action, num_classes=self.domnet.env.action_space.n)
                    obs, reward_log[t], done, info = self.env.step_runner(index, action)
                    with self.queue_lock:
                        self.T += 1
                    t+=1
                    flag = False
                    sys.stdout.flush()
                    time.sleep(0.05)
                print()
                if flag:
                    print("Not calculating loss")
                    continue

                # Accumulate gradients from this run
                bootstrap_value = 0.
                bootstrap_log = dict()
                if not done:
                    bootstrap_value = value_log[t-1]
                bootstrap_log[t] = bootstrap_value
                for i in range(t-1, t_s-1, -1):
                    bootstrap_value = (self.gamma*bootstrap_value) + reward_log[i]
                    bootstrap_log[i] = bootstrap_value
                # with tf.contrib.summary.always_record_summaries():
                #     print("Recording summary")
                #     tf.contrib.summary.scalar('value_error', value_log[t-1])
                #     tf.contrib.summary.scalar('final_reward', tf.constant(reward_log[t-1]))
                #     tf.contrib.summary.scalar('episode_reward', bootstrap_value)

            if len(state_log) == 0:
                print("No states found for entire batch")
                continue

            # compute gradients wrt used weights
            merged_state = [np.concatenate([state[i] for state in list(state_log.values())], axis=0)
                    for i in range(4)]
            tstart = time.time()
            bootstrap_input = np.expand_dims(np.array(list(bootstrap_log.values())[1:]), axis=-1)
            action_input = np.array(list(action_log.values()), dtype=np.int32)
            feed_list = [bootstrap_input] + merged_state + [action_input]
            feed_dict ={self.model.inputs[i].name[:-2]: feed_list[i] for i in range(len(self.model.inputs))}
            feed_dict_logs ={self.model.inputs[i].name : feed_list[i] for i in range(len(self.model.inputs))}
            # grad_update = self.domnet.sess.run([tf.gradients(self.model.output, self.domnet.model.trainable_weights)], feed_dict=feed_dict)
                # exit if kill flag is set
            with self.domnet.weights_lock:
                    # print(feed_dict)
                    self.model.fit(feed_dict)
                    summary = self.domnet.sess.run(self.merged, feed_dict=feed_dict_logs)
                    self.writer.add_summary(summary, self.episode_count)
                    self.episode_count += 1
                    # print(grad_update)
                    # self.grad_queue.put(grad_update, block=False)
                    if self.kill_flags[index]: return
            print("Time taken: ", time.time() - tstart)
            time.sleep(0.02)

    def configure(self, *args, **kwargs):
        self.domnet.configure(*args, **kwargs)
        self.create_model()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.domnet.sess = tf.Session(graph=self.domnet.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        self.merged = tf.summary.merge_all()
        self.domnet.sess.run(tf.global_variables_initializer())
        for thread in self.domnet.update_threads:
            thread.start()

if __name__ == "__main__":
    a3c = A3C()
