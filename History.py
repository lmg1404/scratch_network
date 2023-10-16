import matplotlib.pyplot as plt

class History:
    def __init__(self, loss_string : str):
        self.loss_name = loss_string
        self.train_metric = []
        self.epochs = []
        
    def add_epoch(self, e):
        self.epochs.append(e)
        
    def add_train_metric(self, tm):
        self.train_metric.append(tm)
        
    def plot(self):
        plt.plot(self.epochs, self.train_metric)
        
        plt.xlabel("Epochs")
        plt.ylabel(self.loss_name)
        plt.figure(figsize=(10,6))
        plt.show()