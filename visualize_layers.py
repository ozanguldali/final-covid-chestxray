import torch

from ae.autoencoder import conv_ae
from cnn import device, summary, ae
from cnn.dataset import inv_normalize_tensor, normalize_tensor
from cnn.models import proposednet
from cnn.helper import set_dataset_and_loaders

from torch import nn as nn
import torchvision.models as models

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def show_layer(img, title, w, h):
    fig = plt.figure(figsize=(w, h))
    fig.suptitle(title)
    gs = gridspec.GridSpec(w, h)
    gs.update(wspace=0.025, hspace=0.025)
    for c in range(img.shape[0]):
        ax = plt.subplot(gs[c])
        plt.axis('off')
        c_img = img[c:c + 1, :, :]
        fig.add_subplot(ax)
        plt.imshow(c_img[0].detach().numpy(), interpolation='gaussian')


def visualize(model_name, dataset_folder="dataset", img_size=112, normalize: object = False):
    _, train_loader, _, test_loader = set_dataset_and_loaders(dataset_folder, batch_size=1,
                                                   img_size=img_size, num_workers=4, normalize=normalize)

    original_labels = list(test_loader.dataset.class_to_idx.keys())
    print(test_loader.dataset.class_to_idx)

    show = {
        original_labels[0]: False,
        original_labels[1]: True,
        original_labels[2]: False,
        original_labels[3]: False
    }
    labels = list(show.keys())

    for image, label in train_loader:

        image = image.to(device)
        label = label.to(device)
        label = labels[label.item()]

        if not show[labels[0]] and not show[labels[1]] and not show[labels[2]] and not show[labels[3]]:
            break

        if not show[label]:
            pass

        else:
            if normalize is not False:
                plt.title("original - " + label)
                plt.imshow(inv_normalize_tensor(image[0]).permute(1, 2, 0).detach().numpy(), interpolation='nearest')
                plt.show()
                plt.title("normalized - " + label)
                plt.imshow(image[0].permute(1, 2, 0).detach().numpy(), interpolation='nearest')
                plt.show()

            else:
                plt.title("original - " + label)
                plt.imshow(image[0].permute(1, 2, 0).detach().numpy(), interpolation='nearest')
                plt.show()
                plt.title("normalized - " + label)
                plt.imshow(normalize_tensor(image[0], norm_value=None).permute(1, 2, 0).detach().numpy(), interpolation='nearest')
                plt.show()
            # writer.add_image(tag="initial", img_tensor=image[0])

            if model_name == proposednet.proposednet.__name__:
                model = proposednet.proposednet()
                # summary.get_summary(model, test_loader)
                image = ae(image)

                image = nn.Sequential(*[model.features[i] for i in range(2)])(image)
                # show_layer(image[0], "conv1_1 - " + label, 8, 8)

                image = nn.Sequential(*[model.features[i] for i in range(2, 5)])(image)
                show_layer(image[0], "conv1_2 - " + label, 8, 8)

                image = nn.Sequential(*[model.features[i] for i in range(5, 7)])(image)
                # show_layer(image[0], "conv2_1 - " + label, 12, 12)

                image = nn.Sequential(*[model.features[i] for i in range(7, 11)])(image)
                show_layer(image[0], "conv2_2 - " + label, 12, 12)

                image = nn.Sequential(*[model.features[i] for i in range(11, 13)])(image)
                # show_layer(image[0], "conv3_1 - " + label, 16, 16)

                image = nn.Sequential(*[model.features[i] for i in range(13, 15)])(image)
                # show_layer(image[0], "conv3_2 - " + label, 16, 16)

                image = nn.Sequential(*[model.features[i] for i in range(15, 19)])(image)
                show_layer(image[0], "conv3_3 - " + label, 16, 16)

                image = nn.Sequential(*[model.features[i] for i in range(19, 21)])(image)
                # show_layer(image[0], "conv4_1 - " + label, 24, 24)

                image = nn.Sequential(*[model.features[i] for i in range(21, 23)])(image)
                # show_layer(image[0], "conv4_2 - " + label, 24, 24)

                image = nn.Sequential(*[model.features[i] for i in range(23, 27)])(image)
                show_layer(image[0], "conv4_3 - " + label, 24, 24)

                image = nn.Sequential(model.avgpool, model.flatten)(image)

                image = model.fc1(image)
                image_reshape = image.reshape(shape=(1, 64, 4, 4))
                show_layer(image_reshape[0], "fc1 - " + label, 8, 8)

                image = model.fc2(image)
                image_reshape = image.reshape(shape=(1, 64, 4, 4))
                show_layer(image_reshape[0], "fc2 - " + label, 8, 8)

                y = model.fc3(image)
                print(y)
                prediction = torch.argmax(y, dim=1).item()
                prediction_label = list(show.keys())[prediction]
                print(prediction_label)

            if model_name == "ae":
                image = ae(image)
                show_layer(image[0], "conv1_2 - " + label, 8, 8)

            elif model_name == models.vgg16.__name__:
                model = models.vgg16()
                summary.get_summary(model, train_loader=test_loader)

                image = nn.Sequential(*[model.features[i] for i in range(5)])(image)
                show_layer(image[0], "block1 - " + label, 8, 8)

                image = nn.Sequential(*[model.features[i] for i in range(5, 10)])(image)
                show_layer(image[0], "block2 - " + label, 12, 12)

                image = nn.Sequential(*[model.features[i] for i in range(10, 17)])(image)
                show_layer(image[0], "block3 - " + label, 16, 16)

                image = nn.Sequential(*[model.features[i] for i in range(17, 24)])(image)
                show_layer(image[0], "block4 - " + label, 24, 24)

                image = nn.Sequential(*[model.features[i] for i in range(24, 31)])(image)
                show_layer(image[0], "block5 - " + label, 24, 24)

            plt.show()
            show[label] = False


if __name__ == '__main__':
    visualize("proposednet", img_size=224, normalize=True)
