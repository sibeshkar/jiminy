from jiminy.utils import create_directory
import time
import queue
import threading
import logging
import tensorflow as tf
# import utils
import sys
# import logging

logging.basicConfig(level=logging.DEBUG,
        filename="app.log",
        mode="w", format="%(name)s - %(levelname)s - %(message)s")

class A3C(object):
    def __init__(self,
            gamma=0.99,
            momentum=0.999,
            learning_rate=1e-3,
            entropy_beta=1e-3,
            clip_grad=40.,
            logdir='./logs/a2c',
            batch_size=32):
        create_directory(logdir)
        self.logdir = logdir
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_beta = entropy_beta
        self.clip_grad = clip_grad
        self.momentum = momentum
        self.queue_lock = threading.Lock()
        self.grad_queue = queue.Queue()
        self.batch_size = batch_size
        logging.debug("A3C object created")

        self.action_input = tf.keras.layers.Input(shape=[3], dtype=tf.int32,name='action_input')
        self.reward_input = tf.keras.layers.Input(shape=[1], dtype=tf.int32, name='reward_input')

    @classmethod
    def from_config(cls, config):
        return cls(gamma=config["gamma"], momentum=config["momentum"],
            learning_rate=config["learning_rate"], entropy_beta=config["entropy_beta"],
            clip_grad=config["clip_grad"], logdir=config["logdir"])

    def learn(self, domnet):
        assert (not domnet is None), "Expected domnet object to not be None, got: {}".format(domnet)
        logging.debug("Started A3C learner")

        writer = tf.contrib.summary.create_file_writer(self.logdir)
        writer.set_as_default()

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.momentum)
        self.domnet = domnet
        self.domnet.load(self.logdir + "a3c.h5")
        self.env = domnet.env
        self.n_workers = self.env.n
        self.T = 0
        self.global_step = tf.train.get_or_create_global_step()
        self.kill_flags = [False for _ in range(self.n_workers)]
        self.thread_list = []
        for i in range(self.n_workers):
            logging.debug("Starting worker {} implementing A2C".format(i))
            t = threading.Thread(target=self.runner_wrapper, args=(i,))
            t.start()
            self.thread_list.append(t)
        self.update_thread = threading.Thread(target=self.update_weights, args=())
        self.update_thread.start()
        other_fn = {
                self.save : 100,
                self.discount_t(0.99) : 1000
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
        with tf.device("/gpu:0"):
            try:
                while True:
                    grad_updates = self.grad_queue.get()
                    grad_updates = zip(grad_updates, self.domnet.model.trainable_weights)
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
        with tf.device("/gpu:1"):
            self.runner(*args, **kwargs)

    def runner(self, index, episode_max_length=100, max_step_count=1e6):
        t,t_s = 0,0
        episode = 0
        action_reset = self.env.env._index(5,10 + 75)
        while True:
            grads_policy, grads_value = None, None
            with tf.GradientTape() as tape:
                for _ in range(self.batch_size):
                    episode += 1
                    # noise before running a new episode
                    reward_log = dict()
                    value_log = dict()
                    action_log_prob_log = dict()
                    grads_value, grads_policy = None, None
                    with self.queue_lock:
                        self.global_step.assign_add(1)
                        # exit if the process has run for maximum number of episodes
                        if self.T >= max_step_count: return
                        t_s = t
                    obs = self.env.reset_runner(index)
                    action = self.env.action_space.sample()
                    while obs is None:
                        time.sleep(0.05)
                        obs, _, _, _ = self.env.step_runner(index, action)
                    print("Starting episode: {}".format(episode))
                    self.env.step_runner(index, action_reset)
                    done,value = False, None

                    # synchoronize model from master
                    model = self.domnet.get_runner_instance()
                    flag = True

                    # run episode for max_length time steps
                    while (not done) and t - t_s < episode_max_length:
                        if obs is None:
                            if t - t_s == 0:
                                print("Timed-out", end="")
                            else:
                                print("Ending episode cuz of error", end="")
                            break
                        state, value, action, action_log_prob = self.domnet.step_runner(index, obs, model)
                        if value is None:
                            obs, _, done, info = self.env.step_runner(index, action_reset)
                            print(obs.objects)
                            break
                        value_log[t], action_log_prob_log[t] = value, action_log_prob
                        obs, reward_log[t], done, info = self.env.step_runner(index, action)
                        with self.queue_lock:
                            self.T += 1
                        if not done:
                            print(".", end="")
                        t+=1
                        flag = False
                        sys.stdout.flush()
                    print()
                    if flag:
                        continue

                    # Accumulate gradients from this run
                    bootstrap_value = tf.constant(0.)
                    # val = action_log_prob_log[t-1]*(tf.constant(reward_log[t-1]) - value_log[t-1])
                    if not done:
                        bootstrap_value = tf.stop_gradient(value_log[t-1])
                    for i in range(t-1, t_s-1, -1):
                        bootstrap_value = (self.gamma*bootstrap_value) + tf.constant(reward_log[i])
                        loss_policy = -1.*action_log_prob_log[i]*tf.stop_gradient(bootstrap_value - value_log[i])
                        loss_value = tf.square(tf.stop_gradient(bootstrap_value) - value_log[i])
                        if grads_policy is None: grads_policy = loss_policy
                        else: grads_policy += loss_policy
                        if grads_value is None: grads_value = loss_value
                        else: grads_value += loss_value
                    loss = grads_policy + grads_value
                    print("Epsiode loss: {}, reward: {}".format(loss.numpy(), reward_log[t-1]))
                    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                        tf.contrib.summary.scalar('global_step', self.global_step)
                        tf.contrib.summary.scalar('value_error', loss_value)
                        tf.contrib.summary.scalar('final_reward', tf.constant(reward_log[t-1]))
                        tf.contrib.summary.scalar('episode_reward', bootstrap_value)

            # compute gradients wrt used weights
            with tf.device("/gpu:0"):
                tstart = time.time()
                grad_update = tape.gradient(loss, model.trainable_weights)
                # exit if kill flag is set
                with self.queue_lock:
                    self.grad_queue.put(grad_update, block=False)
                    if self.kill_flags[index]: return
            print("Time taken: ", time.time() - tstart)

if __name__ == "__main__":
    a3c = A3C()
