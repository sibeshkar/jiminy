from jiminy.utils import create_directory
import time
import queue
import threading
import logging
import tensorflow as tf

class A3C(object):
    def __init__(self,
            gamma=0.99,
            momentum=0.999,
            learning_rate=1e-4,
            entropy_beta=1e-3,
            clip_grad=0.05,
            logdir='./logs/a2c'):
        create_directory(logdir)
        self.logdir = logdir
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_beta = entropy_beta
        self.queue_lock = threading.Lock()
        self.grad_queue = queue.Queue()
        logging.debug("A3C object created")

    @classmethod
    def from_config(cls, config):
        return cls(gamma=config["gamma"], momentum=config["momentum"],
            learning_rate=config["learning_rate"], entropy_beta=config["entropy_beta"],
            clip_grad=config["clip_grad"], logdir=config["logdir"])

    def learn(self, domnet):
        assert (not domnet is None), "Expected domnet object to not be None, got: {}".format(domnet)
        logging.debug("Started A3C learner")
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
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

    def update_weights(self):
        while True:
            grad_updates = self.grad_queue.get()
            assert "policy" in updates, "Expected to get updates for policy weights, but got {}".format(updates)
            assert "value" in updates, "Expected to get updates for value weights, but got {}".format(updates)
            self.domnet.model.apply_grads(grad_updates)

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

    def runner(self, index, episode_max_length=100, max_step_count=1e6):
        # model = load_model
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
            model = self.domnet.get_runner_instance()

            # run episode for max_length time steps
            while (not done) and t - t_s < episode_max_length:
                state,value_log[t],action,action_log_prob_log[t] = self.domnet.step_runner(index, obs, model)
                obs, reward_log[t], done, info = self.betadom.step_runner(index, action)
                with self.queue_lock:
                    self.T += 1
                t+=1

            # Accumulate gradients from this run
            bootstrap_value = tf.constant(0.)
            if not done:
                boostrap_value = value
            for i in range(t, t_s, -1):
                bootstrap_value = (self.gamma*boostrap_value) + reward_log[i]
                loss_policy = action_log_prob_log[i]*(tf.stop_gradient(bootstrap_value) - value_dict[i])
                grad_value = tf.square(tf.stop_gradient(bootstrap_value) - value_dict[i])
                if grads_policy is None: grads_policy = grad_policy
                else: grads_policy += grad_policy
                if grads_value is None: grads_value = grad_value
                else: grads_value += grad_value

            grad_update = self.opt.get_gradients(grads_value + grads_policy,
                    model.trainable_weights)

            # push gradients to a queue which updates them
            # exit if kill flag is set
            with self.queue_lock:
                self.grad_queue.put(grad_update, block=False)
                if self.kill_flags[index]: return

if __name__ == "__main__":
    a3c = A3C()
