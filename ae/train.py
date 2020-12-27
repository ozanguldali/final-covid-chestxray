from tqdm.notebook import tqdm, trange

from ae import device

from util.logger_util import log
from util.tensorboard_util import writer


def train_model(model, train_loader, metric, optimizer, num_epochs=25):
    total_loss_history = []

    log.info("Training the model")
    # Iterate through train set mini batches
    for epoch in trange(num_epochs):
        loss_history = []

        for e, (images, _) in enumerate(tqdm(train_loader)):
            # zero the parameter gradients
            optimizer.zero_grad()

            inputs = images.to(device)

            # Do the forward pass
            outputs = model(inputs)

            loss = metric(outputs, inputs)
            loss_history.append(loss.item() * inputs.size(0))

            # Calculate gradients and step
            loss.backward()
            optimizer.step()

        log.info("\nIteration number on epoch %d / %d is %d" % (epoch + 1, num_epochs, len(loss_history)))

        epoch_loss = sum(loss_history) / len(loss_history)
        writer.add_scalar("AE/Loss/Train", epoch_loss, epoch)

        total_loss_history.append(epoch_loss)

        log.info("Epoch {} --> training loss: {}".format(epoch + 1, round(epoch_loss, 4)))

    log.info("\nTotal training iteration: %d" % len(total_loss_history))

    total_loss = sum(total_loss_history) / len(total_loss_history)
    log.info("Average --> training loss: {}" .format(round(total_loss, 6)))
