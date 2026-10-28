from Actions.action import Action
import pytesseract
import time
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
import numpy as np
from dotenv import load_dotenv
import os
from global_vars import GlobalVars

# Load the .env file
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')
ANTIALIAS_METHOD = getattr(Image, os.getenv('ANTIALIAS_METHOD'))


# You can now use the preprocessed image in your existing code

class ExtractTextAction(Action):
    def __init__(self, image_path="test.png", description="", aggregate=False, delay=0, retard=0):
        self.image_path = image_path
        self.description = description
        self.aggregate = aggregate
        self.delay = delay
        self.retard = retard

    def preprocess_image(self, image_path):
        # Open the image file
        img = Image.open(image_path)

        # Convert to grayscale
        img = img.convert('L')

        width, height = img.size
        img = img.resize((width * 5, height * 5), ANTIALIAS_METHOD)
        # Binarization
        if "Q" in self.description:
            img = img.point(lambda x: 0 if x < 140 else 255, '1')
        elif self.description == "marchcount":
            img = img.point(lambda x: 0 if x < 160 else 255, 'L')
            img = ImageOps.invert(img)
            img = ImageOps.expand(img, border=14, fill=255)
        else:
            img = img.point(lambda x: 0 if x < 195 else 255, '1')
            img = ImageOps.invert(img)

        # Save the processed image
        img.save("testprocessed.png")

        return img

    def execute(self):
        img = self.preprocess_image(self.image_path)
        try:
            text = pytesseract.image_to_string(
                img,
                lang='eng',
                config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/"
            )
            print(text.strip())
            if self.description == "marchcount":
                digits = [int(ch) for ch in text if ch.isdigit()]
                if len(digits) >= 2:
                    cur_m, max_m = digits[0], digits[-1]
                    if cur_m < max_m:
                        print(f"March not full ({cur_m}/{max_m})")
                        return True
                    else:
                        print(f"March full ({cur_m}/{max_m})")
                        return False
                elif int(text[0]) < int(text[2]):
                    print("March not full")
                    return True
                else:
                    print("March full")
                    return False
        except Exception:
            return True
        text = pytesseract.image_to_string(img, lang='eng', config='--oem 3 --psm 6 -c tessedit_char_blacklist=|')
        text = text.replace("\n", "")
        print(text)
        if self.description == "Q":
            GlobalVars().Q = text.replace(",", "").replace("\"", "")
        if self.description == "A":
            GlobalVars().A = text.replace(",", "").replace("\"", "")
        if self.description == "B":
            GlobalVars().B = text.replace(",", "").replace("\"", "")
        if self.description == "C":
            GlobalVars().C = text.replace(",", "").replace("\"", "")
        if self.description == "D":
            GlobalVars().D = text.replace(",", "").replace("\"", "")

        return True
