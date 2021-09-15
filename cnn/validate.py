import torch
from tqdm.notebook import tqdm

from cnn import device, ROOT_DIR, MODEL_NAME
from cnn.model import SAVE_FILE
from cnn.save import save_model
from cnn.util import is_verified
from util.file_util import path_exists

from util.logger_util import log
from util.tensorboard_util import writer


def validate_model(model, test_loader, metric, iterator, save):
    correct = 0
    total = len(test_loader.dataset)

    # set the model into evaluation mode
    model = model.eval()
    metric = metric.eval()

    # behavior of the batch norm layer so that it is not sensitive to batch size
    with torch.no_grad():
        # Iterate through test set mini batches
        for e, (images, labels) in enumerate(tqdm(test_loader)):
            # Forward pass
            inputs = images.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)
            truths = torch.sum((predictions == labels).float()).item()
            correct += truths

            loss = metric(outputs, labels).item()

    val_acc = correct / total
    writer.add_scalar(MODEL_NAME[0] + "/Loss/Validation", loss, iterator)
    writer.add_scalar(MODEL_NAME[0] + "/Acc/Validation", val_acc, iterator)
    log.info("{}th Validation --> Loss: {} - Accuracy: {}"
             .format(iterator,
                     round(loss, 6),
                     round(val_acc, 6)))

    if save and is_verified(100 * val_acc):
        exist_files = path_exists(ROOT_DIR + "/saved_models", SAVE_FILE[0], "contains")

        better = len(exist_files) == 0
        if not better:
            exist_acc = []
            for file in exist_files:
                exist_acc.append(float(file.split("_")[0].replace(",", ".")))
            better = all(100 * val_acc > acc for acc in exist_acc)

        if better:
            save_model(model=model, filename=str(round(100 * val_acc, 2)) + "_" + SAVE_FILE[0])
