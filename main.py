import cv2
import numpy as np
import pypdfium2 as pdfium
from pathlib import Path
from PIL import Image
import argparse

root = "."  # take the current directory as root
[f.unlink() for f in Path(str(Path(root)) + '/temp_pages/').glob("*") if f.is_file()]

pdf = None
pdf_name = ""

parser = argparse.ArgumentParser(description='PDF translator')
parser.add_argument('-f','--file', help='Path to pdf file to translate', required=True)
args = vars(parser.parse_args())


if args['file'].split(".")[1] == "pdf":
    pdf_name = Path(args['file']).stem
    pdf = pdfium.PdfDocument(args['file'])

if pdf is None:
    print("Didn't find any pdfs or those found were not selected, exiting")
    exit()


#ROI = image[y1:y2, x1:x2]
#-------------------------------------------
#|                                         | 
#|    (x1, y1)                             |
#|      ------------------------           |
#|      |                      |           |
#|      |                      |           | 
#|      |         ROI          |           |  
#|      | (Region of interest) |           |   
#|      |                      |           |   
#|      |                      |           |       
#|      ------------------------           |   
#|                           (x2, y2)      |    
#|                                         |             
#|                                         |             
#|                                         |             
#-------------------------------------------


colors = {
        0:  (4, 42, 255),
        1: 	(11, 219, 235),
        2: 	(243, 243, 243),
        3: 	(0, 223, 183),
        4: 	(17, 31, 104),
        5: 	(255, 111, 221),
        6: 	(255, 68, 79),
        7: 	(204, 237, 0),
        8: 	(0, 243, 68),
        9:  (189, 0, 255),
        10: (0, 180, 255),
        11: (221, 0, 186),
        12: (0, 255, 255),
        13: (38, 192, 0),
        14: (1, 255, 179),
        15: (125, 36, 255),
        16: (123, 0, 104),
        17: (255, 27, 108),
        18: (252, 109, 47),
        19: (162, 255, 11)
}

import pdf_translate_doc_layout_analyser as layout_analyser
import pdf_translate_ocr as ocr
import pdf_translate_preprocessor as preprocessor
import pdf_translate_translator as translator
import pdf_translate_text_image_creator as text_img_creator


def translated_text_img(img):
    pimg,bcolor,fcolor = preprocessor.find_image_colors(img)
    text = ocr.ocr_text(preprocessor.preprocess_image(pimg))
    text = translator.translate_text(preprocessor.split_into_sentences(text))
    h,w,_ = img.shape
    img = text_img_creator.create_text_image(text,w,h,bcolor,fcolor)
    return img

def replace_in_page(img,results):
    confidence_scores, labels, boxes, text_labels = layout_analyser.results_interpreter(results)
    to_replace = [] #replace as a second step, as it;s possibel for the ocr to read the same region twice in the edge case the layout detect sees boxes inside other boxes
    for score, label, box in zip(confidence_scores, labels, boxes):
        if label in text_labels and score > 0.7:
            x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3]),
            to_replace.append((translated_text_img(img[y1:y2, x1:x2]),x1,y1,x2,y2))
    for img_patch,x1,y1,x2,y2 in to_replace:
        img[y1:y2, x1:x2] = img_patch #translate text regions
    return img

def detect_layout_debug(img,results): 
    categories = []
    confidence_scores, labels, boxes, text_labels = layout_analyser.results_interpreter(results)
    for score, label, box in zip(confidence_scores, labels, boxes):
        if label not in categories:
            categories.append(label)
        x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3]),
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (36,255,12), 1)
        cv2.putText(img, label + " conf:" + str(round(score,2)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[categories.index(label)], 2)
    return img

def pdf_page_to_img(page):
    image = page.render(scale=4).to_pil()
    # image.save(f"output_{i:03d}.jpg")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

import zoom
i = 18
page = pdf[i]
img = pdf_page_to_img(page)
img = replace_in_page(img,layout_analyser.detect_layout(img))
#img = detect_layout_debug(img,layout_analyser.detect_layout(img))
cv2.imwrite(f'temp_pages/page_{i}.png', img)
zoom.show_my_img(img)

"""
i=0
while True:
    try:
        page = pdf[i]
    except:
        break
    img = pdf_page_to_img(page)
    img = replace_in_page2(detect_layout2(img))
    cv2.imwrite(f'temp_pages/page_{i}.png', img)
    i += 1
    print(f"\033[92m Processed page {i} \033[0m")

images = [Image.open(path) for path in Path(root).glob("temp_pages/*.png")]

images[0].save(pdf_name + "_translated.pdf", "PDF" ,resolution=100.0, save_all=True, append_images=images[1:])

"""
