def get_single_image_results(h, real_labels, pred_box_label, iou_thr):
    r_labels = []  # правильные метки
    p_labels = []  # предсказанные метки
    ious = []  # пересечения

    for i in range(len(pred_box_label[h]['boxes'])):
        j = 0
        while j < len(real_labels[h]['boxes']):
            iou = calc_iou(pred_box_label[h]['boxes'][i], real_labels[h]['boxes'][j])
            if iou > iou_thr:
                ious.append(iou)
                p_labels.append(pred_box_label[h]['labels'][i])
                r_labels.append(real_labels[h]['labels'][j])
                break
            j += 1
    iou_sort = np.argsort(ious)[::1]
    gt_match_idx = []
    pred_match_idx = []  # правильно предсказанные метки
    for idx in range(len(ious)):
        gt_idx = r_labels[idx]
        r_idx = p_labels[idx]
        if gt_idx == r_idx:
            gt_match_idx.append(gt_idx)

    tp_apples = gt_match_idx.count('apple')
    tp_oranges = gt_match_idx.count('orange')
    tp_bananas = gt_match_idx.count('banana')

    fp_oranges = pred_box_label[h]['labels'].count('orange') - gt_match_idx.count('orange')
    fp_apples = pred_box_label[h]['labels'].count('apple') - gt_match_idx.count('apple')
    fp_bananas = pred_box_label[h]['labels'].count('banana') - gt_match_idx.count('banana')

    fn_bananas = real_labels[h]['labels'].count('banana') - gt_match_idx.count('banana')
    if fn_bananas < 0:
        fp_bananas += abs(fn_bananas)
        fn_bananas = 0
    fn_apples = real_labels[h]['labels'].count('apple') - gt_match_idx.count('apple')
    if fn_apples < 0:
        fp_apples += abs(fn_apples)
        fn_apples = 0
    fn_oranges = real_labels[h]['labels'].count('orange') - gt_match_idx.count('orange')
    if fn_oranges < 0:
        p_oranges += abs(fn_oranges)
        fn_oranges = 0
    tp = len(gt_match_idx)
    fp = len(pred_box_label[h]['boxes']) - len(gt_match_idx)
    fn = len(real_labels[h]['boxes']) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}, {'tp_oranges': tp_oranges,
                                                                               'fp_oranges': fp_oranges,
                                                                               'fn_oranges': fn_oranges}, {
               'tp_apples': tp_apples,
               'fp_apples': fp_apples,
               'fn_apples': fn_apples}, {'tp_bananas': tp_bananas,
                                         'fp_bananas': fp_bananas,
                                         'fn_bananas': fn_bananas}