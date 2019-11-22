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
        self.acc_mean_win = None
        self.precision_win = None
        self.recall_win = None
        self.f1_win = None

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
    def plot_acc_mean(self, acc_mean, step):
        self.acc_mean_win = self.vis.line(
            [acc_mean],
            [step],
            win=self.acc_mean_win,
            update='append' if self.acc_mean_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Acc',
                title='Test Accuracy (mean)',
            )
        )
    def plot_precision(self, precision, step):
        self.precision_win = self.vis.line(
            [precision],
            [step],
            win=self.precision_win,
            update='append' if self.precision_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Precision',
                title='Precision',
            )
        )
    def plot_recall(self, recall, step):
        self.recall_win = self.vis.line(
            [recall],
            [step],
            win=self.recall_win,
            update='append' if self.recall_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Recall',
                title='Recall',
            )
        )
    def plot_f1(self, f1, step):
        self.f1_win = self.vis.line(
            [f1],
            [step],
            win=self.f1_win,
            update='append' if self.f1_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='F1',
                title='F1',
            )
        )
