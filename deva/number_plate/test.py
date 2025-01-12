import cv2
import pytesseract

def detect_language(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR with English language configuration
    english_text = pytesseract.image_to_string(gray, lang='eng')
    
    # Perform OCR with Devanagari language configuration
    devanagari_text = pytesseract.image_to_string(gray, lang='hin')

    # Check if Devanagari text is detected
    if any(char.isalpha() for char in devanagari_text):
        return 'Devanagari'

    # Check if English text is detected
    if any(char.isalpha() for char in english_text):
        return 'English'
    

    # If no recognizable characters are found
    return 'Unknown'

# Example usage
image_path = 'nep.png'
language = detect_language(image_path)
print(f"The detected language is: {language}")
