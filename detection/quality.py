def calc_precision_recall(image_results):
    true_positive = image_results[list(image_results.keys())[0]]
    false_positive = image_results[list(image_results.keys())[1]]
    false_negative = image_results[list(image_results.keys())[2]]
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)


def metrics(dataset, model):
    images = []
    for i in range(len(dataset)):
        images.append(dataset[i][0])
    real_labels = []
    for i in range(len(dataset)):
        real_labels.append(dataset[i][1])
    predictions = []
    for i in images:
        predictions.append(predict_top(model.cuda(), i))
    pred_box_label = []
    for i in range(len(predictions)):
        d = {}
        d['labels'] = predictions[i][0]
        d['boxes'] = predictions[i][1]
        pred_box_label.append(d)
    tp_apples = 0
    tp_bananas = 0
    tp_oranges = 0
    fp_apples = 0
    fp_bananas = 0
    fp_oranges = 0
    fn_apples = 0
    fn_bananas = 0
    fn_oranges = 0
    for i in range(len(real_labels)):
        image_results_oranges = get_single_image_results(i, real_labels, pred_box_label, 0.5)[1]
        tp_oranges += image_results_oranges['tp_oranges']
        fn_oranges += image_results_oranges['fn_oranges']
        fp_oranges += image_results_oranges['fp_oranges']

        image_results_apples = get_single_image_results(i, real_labels, pred_box_label, 0.5)[2]
        tp_apples += image_results_apples['tp_apples']
        fn_apples += image_results_apples['fn_apples']
        fp_apples += image_results_apples['fp_apples']

        image_results_bananas = get_single_image_results(i, real_labels, pred_box_label, 0.5)[3]
        tp_bananas += image_results_bananas['tp_bananas']
        fn_bananas += image_results_bananas['fn_bananas']
        fp_bananas += image_results_bananas['fp_bananas']

    precision_bananas = tp_bananas / (tp_bananas + fp_bananas)
    precision_oranges = tp_oranges / (tp_oranges + fp_oranges)
    precision_apples = tp_apples / (tp_apples + fp_apples)

    recall_bananas = tp_bananas / (tp_bananas + fn_bananas)
    recall_oranges = tp_oranges / (tp_oranges + fn_oranges)
    recall_apples = tp_apples / (tp_apples + fn_apples)
    return {'precision_bananas': precision_bananas, 'precision_oranges': precision_oranges,
            'precision_apples': precision_apples, 'recall_bananas': recall_bananas,
            'recall_oranges': recall_oranges, 'recall_apples': recall_apples}