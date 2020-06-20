def predict(model, data):
    model.eval()  # Тест
    with torch.no_grad():
        Y_pred = [model(X_batch) for X_batch, _ in data]
    return np.array(Y_pred)

def score_model(model, metric, data):
    model.eval()  # включаем тест
    scores = 0

    with torch.no_grad():
        for X_batch, Y_label in data:
            # получаем вероятности принадлежности пикселей объекту
            Y_pred = torch.sigmoid(model(X_batch.to(device)))
            # Y_pred>0.5 получаем маску от нейросети по порогу в 0.5 вероятности
            scores += metric(Y_pred>0.5, Y_label.to(device)).mean().item()

    return scores/len(data)






