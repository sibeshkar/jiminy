from jiminy.utils import create_directory
import time
import queue
import threading
import logging
import tensorflow as tf
import utils

class A3C(object):
    def __init__(self,
            gamma=0.99,
            momentum=0.999,
            learning_rate=1e-4,
            entropy_beta=1e-3,
            clip_grad=40.,
            logdir='./logs/a2c'):
        create_directory(logdir)
        self.logdir = logdir
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_beta = entropy_beta
        self.clip_grad = clip_grad
        self.momentum = momentum
        self.queue_lock = threading.Lock()
        self.grad_queue = queue.Queue()
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
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.momentum)
        self.domnet = domnet
        self.betadom = domnet.betadom
        self.n_workers = self.betadom.n
        self.T = 0
        self.kill_flags = [False for _ in range(self.n_workers)]
        self.thread_list = []
        for i in range(self.n_workers):
            logging.debug("Starting worker {} implementing A2C".format(i))
            t = threading.Thread(target=self.runner, args=(i,))
            t.start()
            self.thread_list.append(t)
        self.update_thread = threading.Thread(target=self.update_weights, args=())
        # self.update_thread.start()

    def update_weights(self):
        try:
            while True:
                grad_updates = self.grad_queue.get()
                grad_updates = utils.clip_gradient(grad_updates, self.clip_grad)
                grad_updates = zip(grad_updates, self.domnet.model.trainable_variables)
                print(grad_updates)
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
        domnet.save(self.logdir)
        self.betadom.close()
        logging.debug("Closing A3C master")

    def runner(self, index, episode_max_length=10, max_step_count=1e6):
        t,t_s = 0,0
        while True:
            # noise before running a new episode
            reward_log = dict()
            value_log = dict()
            action_log_prob_log = dict()
            grads_value, grads_policy = None, None
            with self.queue_lock:
                # exit if the process has run for maximum number of episodes
                if self.T >= max_step_count: return
                t_s = t
            obs = self.betadom.env.reset_runner(index)
            done,value = False, None

            # synchoronize model from master
            # model = self.domnet.get_runner_instance()

            with tf.GradientTape() as tape:
                # run episode for max_length time steps
                while (not done) and t - t_s < episode_max_length:
                    state, value_log[t], action,action_log_prob_log[t] = self.domnet.step_runner(index, obs)
                    obs, reward_log[t], done, info = self.betadom.step_runner(index, action)
                    with self.queue_lock:
                        self.T += 1
                    t+=1

                # Accumulate gradients from this run
                bootstrap_value = tf.constant(0.)
                val = action_log_prob_log[t-1]*(tf.constant(reward_log[t-1]) - value_log[t-1])
                if not done:
                    bootstrap_value = tf.stop_gradient(value_log[t-1])
                for i in range(t-1, t_s-1, -1):
                    bootstrap_value = (self.gamma*bootstrap_value) + tf.constant(reward_log[i])
                    loss_policy = action_log_prob_log[i]*(bootstrap_value - value_log[i])
                    loss_value = tf.square(tf.stop_gradient(bootstrap_value) - value_log[i])
                    if grads_policy is None: grads_policy = loss_policy
                    else: grads_policy += loss_policy
                    if grads_value is None: grads_value = loss_value
                    else: grads_value += loss_value
                loss = grads_policy + grads_value

            grad_update = tape.gradient(loss, self.domnet.model.trainable_weights)
            # grad_updates = utils.clip_by_value(grad_update, self.clip_grad)
            grad_updates = zip(grad_update, self.domnet.model.trainable_weights)
            print(grad_updates)
            self.opt.apply_gradients(grad_updates)
            # push gradients to a queue which updates them
            # exit if kill flag is set
            # with self.queue_lock:
            #     self.grad_queue.put(grad_update, block=False)
            #     if self.kill_flags[index]: return

if __name__ == "__main__":
    a3c = A3C()
