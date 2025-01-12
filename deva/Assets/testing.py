import pytesseract
import re

def detect_image_lang(img_path):
    try:
        osd = pytesseract.image_to_osd(img_path)
        script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
        conf = re.search("Script confidence: (\d+\.?(\d+)?)", osd).group(1)
        return script, float(conf)
    except pytesseract.pytesseract.TesseractError:
        return "Unprocessable", 0.0
    except:
        return None, 0.0

script_name, confidence = detect_image_lang("magana.png")
if script_name == "Unprocessable":
    print("The script was not abled to be read. There are seem to be some problems.")
else:
    print(f"The confidence of mine is {script_name} with the value of {confidence}.")