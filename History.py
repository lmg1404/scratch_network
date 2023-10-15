import matplotlib.pyplot as plt

class History:
    def __init__(self, loss_string : str):
        self.loss_name = loss_string
        
    def add_epoch(self, e):
        self.epochs = []
        self.epochs.append(e)
        
    def add_train_metric(self, tm):
        self.train_metric = []
        self.train_metric.append(tm)
        
    def add_val_metric(self, vm):
        self.val_metric = []
        self.val_metric.append(vm)
        
    def add_test_metric(self, test_m):
        self.test_metric = []
        self.test_metric.append(test_m)
        self.test_metric *= len(self.epochs) # this should work since the list should be 1 value that many times
        
    def plot(self, train=True, val=False, test=True):
        if train:
            plt.plot(self.epochs, self.train_metric, '-o', label="Train")
            
        if val:
            plt.plot(self.epochs, self.val_metric, '-o', label="Validation")
            
        if test:
            plt.plot(self.epochs, self.test_metric, '-o', label="Test")
        
        plt.xlabel("Epochs")
        plt.ylabel(self.loss_name)
        plt.figure(figsize=(10,6))
        plt.legend(loc='best')
        plt.show()