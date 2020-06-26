classes = ['apple', 'orange', 'banana']
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace the pre-trained head with a new one (note: +1 because of the __background__ class)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
model.to('cuda')
classes = ['__background__'] + classes
int_mapping = {label: index for index, label in enumerate(classes)}