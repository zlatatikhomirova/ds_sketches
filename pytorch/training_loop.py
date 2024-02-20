from math import log10

def train_one_epoch():
    running_loss = 0.
    last_loss = 0.
    dec = int(log10(len(training_loader)))
    if not dec:
        dec =1
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        if i % dec == dec - 1:
            last_loss = running_loss / dec
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


EPOCHS = 5

best_vloss = 1_000_000.
train_loss = []
valid_loss = []

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    model.train()
    print("train:")
    avg_loss = train_one_epoch(epoch)
    running_vloss = 0.0
    model.eval()
    with torch.inferense_mode():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    valid_loss.append(avg_vloss)
    train:_loss.append(avg_loss)
    
    for metric in metrics:
        # tuple(name, F.loss_fn) : list
    print('AVG LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}.pth'.format( epoch)
        torch.save(model.state_dict(), model_path)