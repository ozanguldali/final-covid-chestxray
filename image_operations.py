import os

from PIL import Image, ImageOps

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))


def main(folder, augment_type):
    for filename in os.listdir(folder):
        if not (filename.startswith("horizontalFlip") or filename.startswith("verticalFlip") or filename.startswith("90Rotate") or filename.startswith("180Rotate") or filename.startswith("270Rotate")):
            image_path = os.path.join(folder, filename)
            image = Image.open(image_path)
            if augment_type == "horizontalFlip":
                image_mirror = image.transpose(Image.FLIP_LEFT_RIGHT)
                image_mirror_path = os.path.join(folder, augment_type + "ped_" + filename)
            elif augment_type == "verticalFlip":
                image_mirror = image.transpose(Image.FLIP_TOP_BOTTOM)
                image_mirror_path = os.path.join(folder, augment_type + "ped_" + filename)
            elif augment_type == "90Rotate":
                image_mirror = image.transpose(Image.ROTATE_90)
                image_mirror_path = os.path.join(folder, augment_type + "d_" + filename)
            elif augment_type == "180Rotate":
                image_mirror = image.transpose(Image.ROTATE_180)
                image_mirror_path = os.path.join(folder, augment_type + "d_" + filename)
            elif augment_type == "270Rotate":
                image_mirror = image.transpose(Image.ROTATE_270)
                image_mirror_path = os.path.join(folder, augment_type + "d_" + filename)
            else:
                print("not a valid augment type: " + augment_type)
                import sys
                sys.exit(1)

            rgb = image_mirror.convert('RGB')
            try:
                rgb.save(image_mirror_path)
            except:
                print(filename)


if __name__ == '__main__':
    # RUN JUST ONE TIME
    pass
    # main(folder=ROOT_DIR + '/dataset/train/Viral-COVID19/', augment_type="horizontalFlip")
    # main(folder=ROOT_DIR + '/dataset/train/Viral-COVID19/', augment_type="verticalFlip")
    # main(folder=ROOT_DIR + '/dataset/train/Viral-COVID19/', augment_type="90Rotate")
    # main(folder=ROOT_DIR + '/dataset/train/Viral-COVID19/', augment_type="180Rotate")
    # main(folder=ROOT_DIR + '/dataset/train/Viral-COVID19/', augment_type="270Rotate")
