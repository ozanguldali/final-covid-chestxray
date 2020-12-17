from torchsummary import summary


def get_summary(model, train_loader):
    im_shape = train_loader.__iter__().next()[0].shape
    summary(model, im_shape[1:])

