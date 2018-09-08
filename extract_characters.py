import os
import os.path
import cv2
import glob
import imutils


def extract(CAPTCHA_IMAGE_FOLDER, OUTPUT_FOLDER):
    # Get a list of all the captcha images we need to process
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
    counts = {}
    
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]

        # Load the image and convert it to grayscale
        image = cv2.imread(captcha_image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Add some extra padding around the image
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image (convert it to pure black and white)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hack for compatibility with different OpenCV versions
        contours = contours[0] if imutils.is_cv2() else contours[1]

        letter_image_regions = []
        for contour in contours:
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

        # If we found more or less than 4 letters, skip the image
        if len(letter_image_regions) != 4:
            continue

        # Sort letters by x coord to match text from left-to-right
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # Save out each letter as a single image
        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

            save_path = os.path.join(OUTPUT_FOLDER, letter_text)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            count = counts.get(letter_text, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, letter_image)

            counts[letter_text] = count + 1


if __name__ == '__main__':
    CAPTCHA_IMAGE_FOLDER = "data/generated_captcha_images"
    OUTPUT_FOLDER = "data/extracted_letter_images"

    extract(CAPTCHA_IMAGE_FOLDER, OUTPUT_FOLDER)