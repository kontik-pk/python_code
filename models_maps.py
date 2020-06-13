
#Для визуализации данных и сохранения модели
def summarize_performance(step, generator, dataset, n_samples=3):
    for i in range(3):
        pyplot.subplot(3, 3, 1 + i)
        pyplot.axis('off')
        for X, y in data_tr:
            X = X.cpu()
            y = y.cpu()
            break
        imshow(X[0])
        pyplot.subplot(3, 3, 1 + 3 + i)
        pyplot.axis('off')
        imshow(y[0])

        pyplot.subplot(3, 3, 1 + 3 * 2 + i)
        pyplot.axis('off')
        z = generator(X.cuda())
        z = z.cpu()
        imshow(z[0])
        filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    filename2 = 'model_%06d.h5' % (step + 1)
    torch.save(generator, filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

#Функция для тензоров
def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.detach().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)

#Тренируем
def train(dataset):
    # Models
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    # Оптимизаторы
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #Функции лоссов
    L1_loss = nn.L1Loss()
    loss = nn.BCELoss()
    for i in range(200):
        for batch_idx, (X_batch, Y_batch) in enumerate(data_tr):
            discriminator_optimizer.zero_grad()
            X_batch = X_batch.cuda()
            Y_batch = Y_batch.cuda()
            #Тренируем дискриминатор
            G_result = generator(X_batch) #Генерируем изображение
            D_result_t = discriminator(X_batch, Y_batch) #Передаем в дискриминатор реальные данные
            D_result_f = discriminator(X_batch, G_result) #Передаем в дискриминатор сгенерированные данные
            D_real_loss = loss(D_result_t, torch.ones_like(D_result_t).cuda()) #считаем лосс на реальных данных
            D_fake_loss = loss(D_result_f, torch.zeros_like(D_result_f).cuda()) #считаем лосс на сгенерированных данных
            D_train_loss = (D_real_loss + D_fake_loss) * 0.5 #итоговая ошибка дискриминатора
            D_train_loss.backward()
            discriminator_optimizer.step()

            #Тренируем генератор

            generator.zero_grad()
            G_result = generator(X_batch) #генерируем карту из снимка
            D_result = discriminator(X_batch, G_result) #передаем в дискриминатор
            #Теперь считаем два лосса - первый на решении дискриминатора и метках, второй на сгенерированной карте и реальной
            #для первого лосса - BCELoss, для второго - L1
            G_train_loss = loss(D_result, torch.ones_like(D_result).cuda()) + L1_loss(G_result, Y_batch)
            G_train_loss.backward()
            generator_optimizer.step()

        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, D_real_loss, D_fake_loss, G_train_loss))
        if (i) % 10 == 0:
            summarize_performance(i, generator, dataset)