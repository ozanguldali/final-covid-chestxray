import torch
from tqdm.notebook import tqdm

from sklearn.metrics import confusion_matrix
from cnn import device, MODEL_NAME

from util.logger_util import log
from util.tensorboard_util import writer


def test_model(model, test_loader, iterator=0):
    correct = 0
    total = len(test_loader.dataset)

    prediction_list, label_list = [], []

    # set the model into evaluation mode
    model = model.eval()

    # behavior of the batch norm layer so that it is not sensitive to batch size
    with torch.no_grad():
        # Iterate through test set mini batches
        for e, (images, labels) in enumerate(tqdm(test_loader)):
            # Forward pass
            inputs = images.to(device)
            labels = labels.to(device)
            y = model(inputs)

            predictions = torch.argmax(y, dim=1)
            prediction_list.extend([p.item() for p in predictions])
            label_list.extend([l.item() for l in labels])

            truths = torch.sum((predictions == labels).float()).item()
            correct += truths

    acc = (correct / total)
    log.info('\nTest accuracy: {}'.format(acc))
    if iterator != 0:
        writer.add_scalar(MODEL_NAME[0] + "/Acc/Validation", acc, iterator)

    conf_matrix = confusion_matrix(label_list, prediction_list)
    log.info("Confusion Matrix:\n_________\n| {} | {} | \n|---|---|\n| {} | {} |\n_________"
             .format(conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]))

    return 100 * acc
