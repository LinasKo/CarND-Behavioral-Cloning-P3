# Set working dir
import os
from src.utils import set_working_dir
root_dir = set_working_dir()
data_dirs = [
    os.path.join(root_dir, "data_mouse_drive_2_laps")
]
models_dir = os.path.join(root_dir, "models")

# Data processing
import csv
from scipy import misc
import numpy as np

# Model training
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Visualisation
from src.utils import visualise_random_images


# Parameters
TOP_CROP_PERCENT = 0.35
SIDE_IMAGE_STEERING_CORRECTION = 0.2


def normalize(image_array):
    assert len(np.shape(image_array)) == 4
    return (image_array.astype(np.int16) - 128) / 128


def crop_top(image_array):
    """
    Crop the top of an image since we only care about the road.
    :param image_array: image_array, a numpy array
    :return: cropped image
    """
    assert len(np.shape(image_array)) == 4
    return image_array[:, int(image_array.shape[1] * TOP_CROP_PERCENT):]


def read_center_image(line, img_dir):
    # Line:  center, left, right, steering, throttle, brake, speed
    image_fname = os.path.join(img_dir, os.path.basename(line[0]))
    img = misc.imread(image_fname)
    label = float(line[3])
    return img, label


def read_left_image(line, img_dir):
    # Line:  center, left, right, steering, throttle, brake, speed
    image_fname = os.path.join(img_dir, os.path.basename(line[1]))
    img = misc.imread(image_fname)
    label = float(line[3]) + SIDE_IMAGE_STEERING_CORRECTION
    return img, label


def read_right_image(line, img_dir):
    # Line:  center, left, right, steering, throttle, brake, speed
    image_fname = os.path.join(img_dir, os.path.basename(line[2]))
    img = misc.imread(image_fname)
    label = float(line[3]) - SIDE_IMAGE_STEERING_CORRECTION
    return img, label


def extract_data():
    images = []
    labels = []

    for data_dir in data_dirs:
        img_dir = os.path.join(data_dir, "IMG")

        with open(os.path.join(data_dir, "driving_log.csv")) as csv_file:
            next(csv_file)  # skip first line
            reader = csv.reader(csv_file)
            for line in reader:
                img, label = read_center_image(line, img_dir)
                images.append(img)
                labels.append(label)

    # Bulk preprocess
    images, labels = np.array(images), np.array(labels)

    images = crop_top(images)
    images = normalize(images)

    flipped_images, flipped_labels = np.flip(images, axis=2), np.negative(labels)
    images = np.concatenate((images, flipped_images))
    labels = np.concatenate((labels, flipped_labels))

    return images, labels


def model_lenet():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(160, 320, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(150, activation="relu"))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer='adam')

    return model


if __name__ == "__main__":

    # # Set up the training
    # model = model_lenet()
    # x_train, y_train = extract_data()
    #
    # # Train the model
    # model.fit(x_train, y_train, batch_size=256, epochs=10, validation_split=0.2, shuffle=True)
    #
    # # Save the model
    # model.save(os.path.join(models_dir, "model.h5"))

    images, labels = extract_data()
    visualise_random_images(images, labels)

