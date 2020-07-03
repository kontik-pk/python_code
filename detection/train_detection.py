#Параметры сети изменять в зависимости от данных
def train(dataset, val_dataset=None, epochs=10):
    if not isinstance(dataset, DataLoader):
        dataset = DataLoader(dataset, shuffle=True)
    if val_dataset is not None and not isinstance(val_dataset, DataLoader):
        val_dataset = DataLoader(val_dataset)
    losses = []
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch + 1, epochs))
        model.train()
        for images, targets in dataset:
            _convert_to_int_labels(targets)
            images, targets = _to_device(images, targets)
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print('train loss', total_loss.item())

        if val_dataset is not None:
            avg_loss = 0
            with torch.no_grad():
                for images, targets in val_dataset:
                    _convert_to_int_labels(targets)
                    images, targets = _to_device(images, targets)
                    loss_dict = model(images, targets)
                    total_loss = sum(loss for loss in loss_dict.values())
                    avg_loss += total_loss.item()

            avg_loss /= len(val_dataset.dataset)
            losses.append(avg_loss)
        print('val_loss', avg_loss)
        filename = 'model_%06d.pth' % (epoch)
        torch.save(model, filename)
        lr_scheduler.step()
    if len(losses) > 0:
        return losses
