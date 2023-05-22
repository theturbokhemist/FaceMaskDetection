import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.models as models

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt



class FinetuneInceptionV3():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_path = 'inceptionv3_model'
        self.criterion = nn.CrossEntropyLoss()

        
    def finetune_model(self, x_train, y_train, x_cv, y_cv, x_test, y_test,
                       model_path, lab2idx, batch_size=32, n_epochs=20):
        if model_path:
            self.model_path = model_path
            
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
            
        self.init_model(num_classes=len(lab2idx))
            
        self.model.train()
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        train_losses = []
        cv_losses = []

        best_acc = 0

        for epoch in range(n_epochs):
            print('\nEpoch: %d' % epoch)

            idx = torch.randperm(x_train.size(0))
            x_train = x_train[idx]
            y_train = y_train[idx]

            ep_ttl = 0

            for j in range(0, x_train.size(0), batch_size):
                ep_ttl += 1
                x_batch = x_train[j:j+batch_size].to(self.device)
                y_batch = y_train[j:j+batch_size].to(self.device)

                #reset gradient for this iteration
                self.optimizer.zero_grad()

                #run the data through the model
                output = self.model(x_batch)

                #get the negative log likelihood loss
                loss = self.criterion(output, y_batch)

                #calculate the gradients
                loss.backward()

                #update the parameters from the gradients
                self.optimizer.step()

                if ep_ttl%10==0:
                    print('Epoch: %d, Batch: %d, Loss: %.6f' % (epoch, j, loss.item()))
                    train_losses.append(loss.item())

            ##### more testing, used to be at epoch level
            print('Testing model...')
            acc, cv_loss = self.test(x_cv.to(self.device), y_cv.to(self.device), lab2idx)
            scheduler.step(cv_loss)
            cv_losses.append(cv_loss)
            print('CV accuracy %.6f, prev best acc: %.6f %s' % (acc, best_acc, '!! IMPROVED !!' if acc>best_acc else ''))

            if acc>best_acc:
                best_acc = acc
                no_improvement = 0
                print('Saving model...')
                torch.save(self.model.state_dict(), '%s/model.pt' % self.model_path)
                torch.save(self.optimizer.state_dict(), '%s/optimizer.pt' % self.model_path)
            else:
                no_improvement += 1

            if no_improvement >= 50:
                print('no improvement in several epochs, breaking')
                break

        self.model.load_state_dict(torch.load('%s/model.pt' % self.model_path))
        test_acc, _ = self.test(x_test.to(self.device), y_test.to(self.device), lab2idx, True)
        print('final test accuracy: %.6f' % test_acc)
        
        fig = plt.figure()
        plt.plot(train_losses, color='blue')
        plt.plot(cv_losses, color='orange')
        plt.legend(['Train(blue) & CrossVal(orange) Loss'], loc='upper right')
        
        self.model.eval()

        return self.model, train_losses, cv_losses
    
    
        
    def init_model(self, num_classes):
        #inception v3 has auxillary outputs we don't want
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        self.model.fc = nn.Linear(2048, num_classes)
        
        # apply differential learning rates
        self.optimizer = optim.AdamW([
            {'params': self.model.fc.parameters(), 'lr': 1e-3},
            {'params': self.model.Mixed_7c.parameters(), 'lr': 1e-4},
            {'params': self.model.Mixed_7b.parameters(), 'lr': 1e-4},
            {'params': self.model.Mixed_7a.parameters(), 'lr': 1e-5},
            {'params': self.model.Mixed_6e.parameters(), 'lr': 1e-5},
            {'params': self.model.Mixed_6d.parameters(), 'lr': 1e-5},
            {'params': self.model.Mixed_6c.parameters(), 'lr': 1e-6},
            {'params': self.model.Mixed_6b.parameters(), 'lr': 1e-6},
            {'params': self.model.Mixed_6a.parameters(), 'lr': 1e-6},
            {'params': self.model.Mixed_5d.parameters(), 'lr': 1e-7},
            {'params': self.model.Mixed_5c.parameters(), 'lr': 1e-7},
            {'params': self.model.Mixed_5b.parameters(), 'lr': 1e-7}
        ], lr=1e-8)
        
        
    def test(self, x_test, y_test, lab2idx, print_report=False):
        self.model.eval()

        correct = 0
        loss = 0
        with torch.no_grad():
            output = self.model(x_test)

            loss = self.criterion(output, y_test)

            #select the indices of the maximum output values/prediction
            _, y_pred = torch.max(output, 1)

            #compare them with the target digits and sum correct predictions
            correct = y_pred.eq(y_test).sum()

        acc = correct / y_test.size()[0]

        print('Accuracy %.6f, %d of %d' % (acc, correct, y_test.size(0)))

        if print_report:
            idx2lab = {v:k for k,v in lab2idx.items()}
            class_labels = [idx2lab[i] for i in range(len(idx2lab))]

            print('\n\n')
            print(classification_report(y_test.tolist(), y_pred.tolist(), target_names=class_labels, digits=4))
            print('\n\n')

            cm = confusion_matrix(y_test.tolist(), y_pred.tolist())
            fig, ax = plt.subplots(figsize=(12,10))
            f = sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
            
        self.model.train()

        return acc, loss.item()
        
                       
                       
                       
                       
                       