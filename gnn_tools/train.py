import time
import numpy as np
import matplotlib.pyplot as plt
import torch

def train(model, loader, cuda, criterion, optimizer):
    model.train()
    running_loss=0
    for data_set in loader: # Iterate in batches over the training dataset.
        data_set = data_set[:9]
        for data in data_set:
            # zero the parameter gradients
            data.to(cuda)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.u, data.pseudo_mH, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y.view(-1,1))  # Compute the loss per entry.
            loss = torch.div(torch.dot(loss.flatten(), data.w.flatten()), 100*len(data.w.flatten()))
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            running_loss += loss.item()
            print(out.item(), data.y.item(), loss.item())
    return running_loss

def get_metrics(model, loader, cuda, criterion):
     model.eval()
     running_loss=0
     for data_set in loader:
        data_set = data_set[:9]
        for data in data_set:  # Iterate in batches over the training/test dataset.
            data.to(cuda)
            # zero the parameter gradients
            out = model(data.x, data.edge_index, data.edge_attr, data.u, data.pseudo_mH, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y.view(-1,1))  # Compute the loss per entry.
            loss = torch.div(torch.dot(loss.flatten(), data.w.flatten()), 100*len(data.w.flatten()))
            running_loss += loss.item()
     return running_loss


def runTraining(model, loader_train, loader_test, patience, max_epochs, modelname, cuda, criterion, optimizer):
#  start=time.time()
  epochs=[]
  train_losses=[]
  test_losses=[]
  train_aucs=[]
  test_aucs=[]
  min_loss=np.inf
  no_improvement_steps=0
  for epoch in range(1, max_epochs):
      print(f'Computing epoch {epoch} out of {max_epochs}')
##      epoch_start=time.time()
      av_loss=train(model, loader_train, cuda, criterion, optimizer)
##      epoch_end=time.time()
      #train_loss=get_metrics(loader_train)
      test_loss=get_metrics(model, loader_test, cuda, criterion)
      #test_auc=get_auc_epoch(loader_test)
      #train_auc=get_auc_epoch(loader_train)
##      metrics_end=time.time()
##      elapsed=time.time()-start
#      print('Epoch: {}, \tTraining loss: {:.4f},\tTest loss: {:.4f}, \t\t\tTraining time: {:.2f}s, \tEvaluation time: {:.2f}s, \tTotal time elapsed: {:.2f}s'.format(epoch, av_loss, test_loss, epoch_end-epoch_start, metrics_end-epoch_end, elapsed))
      # Early-stopping / checkpointing 
      if test_loss<min_loss: # if loss is lower than previous best
          no_improvement_steps=0 # reset patience counter
          print('Test loss improved from {:.4f} to {:.4f}. Saving model to "bestModel.pt"'.format(min_loss, test_loss))
          min_loss=test_loss
          torch.save(model, modelname) # save model 
      else:
          print('Test loss did not improve.')
          no_improvement_steps+=1 # else add 1 to the patience counter and dont save
      if no_improvement_steps>=patience:
        print('No improvement for {} epochs. Early stopping now.'.format(no_improvement_steps))
        break
      print('\n')
      epochs.append(epoch)
      train_losses.append(av_loss)
      test_losses.append(test_loss)
      loader_train.index = 0
      loader_test.index = 0
      print(f'Finished computation of epoch {epoch}')

  model = torch.load(modelname)
  train_loss=get_metrics(model, loader_train, cuda, criterion)
  test_loss=get_metrics(model, loader_test, cuda, criterion)
  print('Reloaded best model .\tTrain loss {:.4f},\tTest Loss {:.4f}'.format(train_loss, test_loss))
  
  fig=plt.figure(figsize=(10,8))
  plt.plot(epochs, train_losses, label='Train loss')
  plt.plot(epochs, test_losses, label='Test loss')
  plt.legend(loc='best')
  plt.show()

  return model
