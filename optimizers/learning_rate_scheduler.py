class LearningRateScheduler:
    def __init__(self, initial_lr=0.01, decay_factor=0.1, decay_every=10):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_every = decay_every

    def get_learning_rate(self, epoch):
        if epoch % self.decay_every == 0 and epoch > 0:
            return self.initial_lr * (self.decay_factor ** (epoch // self.decay_every))
        return self.initial_lr
