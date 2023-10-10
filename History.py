import matplotlib.pyplot as plt

class History:
    def __init__(self, loss_string: str):
        self.epochs = []
        self.train_metric = []
        self.val_metric = []
        self.loss_name = loss_string
        
    def add_epoch(self, e):
        self.epochs.append(e)
        
    def add_train_metric(self, tm):
        self.train_metric.append(tm)
        
    def add_val_metric(self, vm):
        self.val_metric.append(vm)
        
    def plot(self, train=True, val=True):
        if train:
            plt.plot(self.epochs, self.train_metric, '-o', label="Train")
            
        if val:
            plt.plot(self.epochs, self.val_metric, '-o', label="Validation")
        
        plt.xlabel("Epochs")
        plt.ylabel(self.loss_name)
        plt.figure(figsize=(10,6))
        plt.legend(loc='best')
        plt.show()