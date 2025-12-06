
from data_loader import creat_dataloader
from model import LenetClassifier
from train3_evalid import train_model,evaluate
import torch.nn as nn
import torch . optim as optim
import torch
import time 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
)
train_dataloader,valid_dataloader,num_classes = creat_dataloader()
lenet_model = LenetClassifier(num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet_model.parameters())
num_epoch = 10
save_model = '/content/drive/MyDrive/Lehaiphong_AI/research/level1'

def run_train_model_lenetmodel(num_classes,num_epoch,device,model,train_dataloader,valid_dataloader, criterion,optimizer):
  
  train_accs,train_losses = [],[]
  eval_accs,eval_losses = [],[]
  best_loss_eval =100 
  best_loss_eval = 100
  model=model 
  for epoch in range(1,num_epoch+1):
    epoch_start_time = time.time()

    train_acc,train_loss,model_final = train_model(model,optimizer,criterion,train_dataloader,device,epoch)
    train_accs.append(train_acc)
    train_losses.append(train_loss)

    eval_acc,eval_loss = evaluate(model_final,criterion,valid_dataloader,device)
    eval_accs.append(eval_acc)
    eval_losses.append(eval_loss)
    if eval_loss < best_loss_eval:
      torch.save(model.state_dict(),save_model + '/lenet_model.pt'
                 )
    print("epoch:",epoch,"eval_acc:",eval_acc,"eval_loss:",eval_loss) 
  return eval_accs,eval_losses,train_accs,train_losses

run_train_model_lenetmodel(num_classes,num_epoch,device,lenet_model,train_dataloader,valid_dataloader, criterion,optimizer)
  
