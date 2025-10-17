from paddleocr import PaddleOCR
from paddleocr import _utils as util
import logging
import warnings

warnings.filterwarnings("ignore")
util.logging.logging.disable(logging.INFO)

ocr = PaddleOCR(use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
                use_doc_unwarping=False, # Disables text image rectification model via this parameter
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec")

def ocr_text(img):
    results = ocr.predict(img)
    text = ""
    for i in range(len(results[0].get("rec_texts"))):
        #coords = results[0].get("dt_polys")[i]
        text = text + results[0].get("rec_texts")[i]
    return text

