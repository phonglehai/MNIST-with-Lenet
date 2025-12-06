
import time
def train_model(model,optimizer,criterion,train_dataloader,device,epoch=0,log_interval=50):
  model.train()
  total_acc,total_count = 0,0
  losses = []
  start_time = time.time()
  for idx, (inputs,labels) in enumerate(train_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    predictions = model(inputs)

    loss = criterion(predictions,labels)
    losses.append(loss.item())

    #backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_( model.parameters () , 0.1)
    optimizer.step()
    total_acc += (predictions.argmax(1) == labels).sum().item()
    total_count += labels.size(0)

    if idx % log_interval ==0 and idx >0:
      elapsed = time.time() -start_time
      print(
          "|epoch {:3d}|{:5d}/{:5d} batches"
          " accurracy{:8.3f}" .format(epoch,idx,len(train_dataloader),total_acc/total_count)
       )
      total_acc , total_count = 0, 0
      start_time = time.time()
  epoch_acc = total_acc/total_count
  epoch_loss = sum(losses)/len(losses)
  return epoch_acc, epoch_loss
def evaluate(model,criterion, valid_dataloader,device):
  model.eval
  total_acc, total_count = 0,0
  losses = []
  with torch.no_grad():
    for idx, (inputs,labels) in enumerate(valid_dataloader):
      inputs = inputs.to(device)
      labels = labels.to(device)

      predictions = model(inputs)

      loss = criterion(predictions , labels)

      losses.append(loss)

      total_acc = (predictions.argmax(1) == labels).sum().item()
      total_count += labels.size(0)

  epoch_acc = total_acc / total_count
  epoch_loss = sum(losses)/len(losses)
  return epoch_acc, epoch_loss


