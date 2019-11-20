import visdom
from datetime import datetime

class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None
        self.acc_win = None

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Train Loss',
            )
        )
    def plot_acc(self, acc, step):
        self.acc_win = self.vis.line(
            [acc],
            [step],
            win=self.acc_win,
            update='append' if self.acc_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Acc',
                title='Test Accuracy',
            )
        )