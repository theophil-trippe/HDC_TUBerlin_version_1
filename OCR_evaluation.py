import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
from PIL import Image, ImageDraw, ImageOps
from PIL import ImageFont


def normalize(img):
    """
    Linear histogram normalization
    """
    arr = np.array(img, dtype=float)

    arr = (arr - arr.min()) * (255 / arr[:, :50].min())
    arr[arr > 255] = 255
    arr[arr < 0] = 0

    return Image.fromarray(arr.astype('uint8'), 'L')


def evaluateImage(imageFile, trueTextFile):
    with open(trueTextFile, 'r') as f:
        trueText = f.readlines()

    # remome \n character
    trueText = [text.rstrip() for text in trueText]

    # load image and convert to grayscale
    img = Image.open(imageFile)
    # img.show()
    img = ImageOps.grayscale(img)
    # img.show()
    w, h = img.size

    img = normalize(img)
    # img.show()

    # resize image to improve OCR
    img = img.resize((int(w / 2), int(h / 2)))
    w, h = img.size

    # run OCR
    options = r'--oem 1 --psm 6 -c load_system_dawg=false -c load_freq_dawg=false  -c textord_old_xheight=0  -c textord_min_xheight=100 -c ' \
              r'preserve_interword_spaces=0'
    OCRtext = pytesseract.image_to_string(img, config=options)

    # debug mode
    showOCRImage = False
    if showOCRImage:
        boxes = pytesseract.image_to_boxes(img, config=options)
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype('./verdanaRef.ttf', 30)
        for b in boxes.splitlines():
            text = b[0]
            b = [int(x) for x in b.split(' ')[1:]]
            draw.rectangle([b[0], h - b[1], b[2], h - b[3]], fill=None, outline='#ff0000', width=1)
            draw.text((b[0], h - b[1]), text, font=font, fill=0)

        img.show()

    # removes form feed character  \f
    OCRtext = OCRtext.replace('\n\f', '').replace('\n\n', '\n')

    # split lines
    OCRtext = OCRtext.split('\n')

    # remove empty lines
    OCRtext = [x.strip() for x in OCRtext if x.strip()]

    # check if OCR extracted 3 lines of text
    print('File:' + imageFile)
    print('True text (middle line): %s' % trueText[1])

    if len(OCRtext) != 3:
        print('ERROR: OCR text does not have 3 lines of text!')
        print(OCRtext)
        return None
    else:
        score = fuzz.ratio(trueText[1], OCRtext[1])
        print('OCR  text (middle line): %s' % OCRtext[1])
        print('Score: %d' % score)

        return float(score)


if __name__ == "__main__":

    InputFile = './example.png'
    trueText = './example_TrueText.txt'

    score = evaluateImage(InputFile, trueText)
