from keras.models import load_model
from utils import resize_to_fit
from imutils import paths
import argparse
import numpy as np
import imutils
import cv2
import pickle


def loadLabels(MODEL_LABELS_FILENAME):
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)
    return lb

def loadModel(MODEL_FILENAME):
    model = load_model(MODEL_FILENAME)
    return model


def resolveLetter(model, lb, letter_bounding_box, image, predictions, output):
    # Grab the coordinates of the letter in the image
    x, y, w, h = letter_bounding_box
    # Extract the letter from the original image with a 2-pixel margin around the edge
    letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
    letter_image = resize_to_fit(letter_image, 20, 20)
    letter_image = np.expand_dims(letter_image, axis=2)
    letter_image = np.expand_dims(letter_image, axis=0)

    prediction = model.predict(letter_image)
    # Convert the one-hot-encoded prediction back to a normal letter
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    # draw the prediction on the output image
    cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
    cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def solveCaptcha(image_file, lb, model):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []
    # Loop through each of the four contours 
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    if len(letter_image_regions) != 4:
        return 'Error! Failed to find 4 distinct characters.'

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    for letter_bounding_box in letter_image_regions:
        resolveLetter(model, lb, letter_bounding_box, image, predictions, output)

    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))


if __name__ == '__main__':
    MODEL_FILENAME = "captcha_model.hdf5"
    MODEL_LABELS_FILENAME = "model_labels.dat"
    CAPTCHA_IMAGE_FOLDER = "data/generated_captcha_images"

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default="data/generated_captcha_images/22A6.png")
    args = parser.parse_args()
    IMAGE_FILE = args.img

    lb = loadLabels(MODEL_LABELS_FILENAME)
    model = loadModel(MODEL_FILENAME)

    solveCaptcha(IMAGE_FILE, lb, model)