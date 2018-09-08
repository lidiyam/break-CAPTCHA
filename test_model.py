from break_captcha import loadModel, loadLabels, solveCaptcha
from imutils import paths
import numpy as np


def testWithRandom(lb, model, CAPTCHA_IMAGE_FOLDER):
    # Grab some random CAPTCHA images to test against.
    captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
    captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

    for image_file in captcha_image_files:
    	print(image_file)
        solveCaptcha(image_file, lb, model):


if __name__ == '__main__':
	MODEL_FILENAME = "captcha_model.hdf5"
    MODEL_LABELS_FILENAME = "model_labels.dat"
    CAPTCHA_IMAGE_FOLDER = "data/generated_captcha_images"
    
    lb = loadLabels(MODEL_LABELS_FILENAME)
    model = loadModel(MODEL_FILENAME)
    testWithRandom(lb, model, CAPTCHA_IMAGE_FOLDER)
