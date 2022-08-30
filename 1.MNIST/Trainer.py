from cProfile import label
import torch
import matplotlib.pyplot as pl
import os

class trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, max_epochs, path , state_dict_name = 'check_point_',  when_to_stop = 5, save_model = True) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.max_epochs = max_epochs
        self.losses_train = []
        self.losses_val = []
        self.device = "cpu"
        self.state_dict_name = state_dict_name
        self.best_val = 1e10
        self.best_train = 1e10
        self.best_optimizer_state = self.optimizer.state_dict()
        self.best_model_state = self.model.state_dict()
        self.stop = 0
        self.when_to_stop = when_to_stop
        self.save_model = save_model
        self.path = path

    def fit(self):

        if torch.cuda.is_available():
            self.device = "cuda:0"
        
        self.model.to(self.device)
        for epoch in range(self.max_epochs):

            for i, batch in enumerate(self.train_loader, 0):

                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()

            train_loss = self.evaluation(self.train_loader)
            val_loss = self.evaluation(self.val_loader)

            self.losses_train.append(train_loss)
            self.losses_val.append(val_loss)
            print(train_loss, " , ", val_loss)

            overfit = self.__overfitting(epoch, val_loss, train_loss)

            if(overfit and self.__early_stopping) :
                print(f'Early Stopping! Overfitting\nBest Validation Loss : {self.best_val}, Current_Loss : {val_loss}')
                return
            if(epoch == self.max_epochs - 1):
                self.__save_checkpoint(epoch, train_loss, val_loss, self.model.state_dict(), self.optimizer.state_dict())

    def evaluation(self, data_loader):

        self.model.eval()
        losses = []
        
        for i, batch in enumerate(data_loader, 0):
            
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
            losses.append(loss.cpu())
        
        data_loss = torch.tensor(losses).mean().item()
        self.model.train()

        return data_loss
    
    def plot(self):

        epochs = list(range(1, len(self.losses_train)+1))
        
        pl.figure(figsize=(5, 2.7), layout='constrained')
        pl.plot(epochs, self.losses_train, label='train_loss')
        pl.plot(epochs, self.losses_val, label='val_loss')
        pl.xlabel('epoch')
        pl.ylabel('loss')
        pl.title("Train/Val Loss")
        pl.legend()
    


    def __save_checkpoint(self, epoch, train_loss, val_loss, model_state_dict, optimizer_state_dict):
        checkpoint = {
            'epoch' : epoch,
            'model_state' : model_state_dict,
            'optimizer_state' : optimizer_state_dict,
            'train_loss' : train_loss,
            'val_loss' : val_loss
        }
        
        if not os.path.exists('checkpoints/'):
            os.mkdir('checkpoints/')
        checkpointPath = 'checkpoints/' + self.path
        if not os.path.exists(checkpointPath):
            os.mkdir(checkpointPath)
        new_path = ''
        for i in range(1000):
            new_path = checkpointPath + self.state_dict_name + str(i) + '.pth'
            if not os.path.exists(new_path):
                torch.save(checkpoint, new_path)
                break
        print('*************************************************************')
        print(f'[Checkpoint: epoch: {epoch+1}, val_loss: {val_loss:.2f} \nsaved on {new_path}]')
        print('*************************************************************')


    def __overfitting(self, epoch, val_loss, train_loss):
        overfit = False
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.best_train = train_loss
            self.best_model_state = self.model.state_dict()
            self.best_optimizer_state = self.optimizer.state_dict()
            self.stop = 0
        else:
            self.stop += 1
        if self.stop >= self.when_to_stop:
            overfit = True
            print("Overfitting!")
            if self.save_model:
                self.__save_checkpoint(epoch - self.stop, self.best_train, self.best_val, self.best_model_state, self.best_optimizer_state)
            if not self.__early_stopping:
                print('Restarting Overfitting Check')
                print('Best Validation: ', self.best_val)
                self.best_train = train_loss
                self.best_val = val_loss
                self.best_model_state = self.model.state_dict()
                self.best_optimizer_state = self.optimizer.state_dict()
                self.stop = 0

        return overfit
    