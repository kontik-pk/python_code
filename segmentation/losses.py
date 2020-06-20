def train(model, opt, loss_fn, epochs, data_tr, data_val, data_ts, show_val=True):
    X_val, Y_val = next(iter(data_val))
    history = []
    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch + 1, epochs))

        avg_loss = 0
        corr = 0
        processed_size = 0
        model.train()
        for X_batch, Y_batch in data_tr:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            opt.zero_grad()
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)
            loss.backward()
            opt.step()

            avg_loss += loss / len(data_tr)
            corr += iou_pytorch(torch.sigmoid(model(X_batch.to(device))) > 0.5, Y_batch.to(device)).mean().item()
            processed_size += X_batch.size(0)
        train_loss = avg_loss
        train_acc = corr / len(data_tr)
        toc = time()
        print('loss: %f' % avg_loss)

        # визуализируем получаемые на каждой эпохе результаты
        model.eval()  # testing mode
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0
        for inputs, labels in data_val:  # Валидация
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.set_grad_enabled(False):
                preds = model(inputs)
                loss = loss_fn(preds, labels)
            running_loss += loss.item() / len(data_val)
            running_corrects += iou_pytorch(torch.sigmoid(model(inputs)) > 0.5, labels).mean().item()
            processed_size += inputs.size(0)
        val_loss = running_loss
        val_acc = running_corrects / len(data_val)

        model.eval()  # Тест
        test_loss = 0.0
        test_running_corrects = 0
        for inputs_ts, labels_ts in data_ts:  # Тест
            inputs_ts = inputs_ts.to(DEVICE)
            labels_ts = labels_ts.to(DEVICE)
            with torch.set_grad_enabled(False):
                preds = model(inputs_ts)
                loss = loss_fn(preds, labels_ts)
            test_loss += loss.item() / len(data_ts)
            test_running_corrects += iou_pytorch(torch.sigmoid(model(inputs_ts)) > 0.5, labels_ts).mean().item()
        ts_loss = test_loss
        ts_acc = test_running_corrects / len(data_ts)

        if show_val:
            # предскажем маску на изображениях из val выборки
            Y_hat_val = torch.sigmoid(model(X_val.to(device))).cpu().detach().numpy()

        else:
            # выполним предсказания на изображениях из train выборки
            Y_hat_train = torch.sigmoid(Y_pred).cpu().detach().numpy()
            X_batch = X_batch.cpu().detach().numpy()
            Y_batch = Y_batch.cpu().detach().numpy()

        # визуализируем получаемые на каждой эпохе результаты
        clear_output(wait=True)

        if show_val:
            row_names = ['Картинка из val', 'Маска нейросети на картинке val',
                         'Реальная маска картинки из val выборки']
        else:
            row_names = ['Картинка из train', 'Маска нейросети на картинке train',
                         'Реальная маска картинки из train выборки']

        fig, big_axes = plt.subplots(figsize=(12, 8), nrows=3, ncols=1, sharey=True)
        for row, big_ax in enumerate(big_axes, start=1):
            big_ax.set_title(row_names[row - 1], fontsize=16)

            big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')

            big_ax._frameon = False

        for k in range(6):
            if show_val:
                ax = fig.add_subplot(3, 6, k + 1)
                ax.imshow(X_val[k, 0], cmap='gray')
                plt.axis('off')
                fig.suptitle('%d / %d - loss: %f' % (epoch + 1, epochs, avg_loss))

                ax = fig.add_subplot(3, 6, k + 7)
                plt.imshow(Y_hat_val[k, 0] > 0.5, cmap='gray')
                plt.axis('off')

                ax = fig.add_subplot(3, 6, k + 13)
                plt.imshow(Y_val[k, 0], cmap='gray')
                plt.axis('off')

            else:
                ax = fig.add_subplot(3, 6, k + 1)
                ax.imshow(X_batch[k, 0], cmap='gray')
                plt.axis('off')
                fig.suptitle('%d / %d - loss: %f' % (epoch + 1, epochs, avg_loss))

                ax = fig.add_subplot(3, 6, k + 7)
                plt.imshow(Y_hat_train[k, 0] > 0.5, cmap='gray')
                plt.axis('off')

                ax = fig.add_subplot(3, 6, k + 13)
                plt.imshow(Y_batch[k, 0], cmap='gray')
                plt.axis('off')

        fig.set_facecolor('w')
        plt.tight_layout()
        plt.show()
        history.append((train_loss, train_acc, val_loss, val_acc, ts_loss, ts_acc))
    return history






