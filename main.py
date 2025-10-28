import cv2
import numpy as np
import pypdfium2 as pdfium
from pathlib import Path
from PIL import Image

root = "."  # take the current directory as root
[f.unlink() for f in Path(str(Path(root)) + '/temp_pages/').glob("*") if f.is_file()]

pdf = None
pdf_name = ""
for path in Path(root).glob("**/*.pdf"):
    answer = input(f"Select {path} for processing y/n?")
    if answer.lower()[0] == "y":
        pdf = pdfium.PdfDocument(path)
        pdf_name = Path(path).stem
        break

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

def replace_in_page(results):
    confidence_scores, labels, boxes, text_labels = layout_analyser.results_interpreter(results)
    for score, label, box in zip(confidence_scores, labels, boxes):
        print(score)
        print(label)
        print(box)
        if label in text_labels:
            x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3]),
            img[y1:y2, x1:x2] = translated_text_img(img[y1:y2, x1:x2]) #translate text regions
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
img = replace_in_page(layout_analyser.detect_layout(img))
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
