import cv2
import numpy as np
import pypdfium2 as pdfium
from pathlib import Path
#from ultralytics import YOLO
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import torch

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

"""
def replace_in_page(results):
    boxes = results[0].boxes
    names = results[0].names
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    clsid = boxes.cls.cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        confidence = conf[i]
        class_id = int(clsid[i])
        class_name = names[class_id]
        
        if (class_name == "Text" or class_name == "Section-header") and confidence > 0.7:
            img[y1:y2, x1:x2] = translated_text_img(img[y1:y2, x1:x2]) #translate text regions
    return img

def detect_layout(img):
    #yolo for image layout detection
    model_path = "yolov12l-doclaynet.pt"
    model = YOLO(model_path)
    #img = preprocessor.preprocess_image(img)
    results = model(img, device="cpu")  # Force CPU
    return results
"""

classes_map = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
    11: "Document Index",
    12: "Code",
    13: "Checkbox-Selected",
    14: "Checkbox-Unselected",
    15: "Form",
    16: "Key-Value Region",
}
text_labels = set(['Caption','List-item','Text','Title','Senction-header', 'Table'])

def replace_in_page2(results):
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = round(score.item(), 2)
            label = classes_map[label_id.item()]
            box = [round(i, 2) for i in box.tolist()]
            if label in text_labels:
                x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3]),
                img[y1:y2, x1:x2] = translated_text_img(img[y1:y2, x1:x2]) #translate text regions
    return img

def detect_layout2(img):
    image = Image.fromarray(img)
    image_processor = RTDetrImageProcessor.from_pretrained("ds4sd/docling-layout-heron")
    model = RTDetrV2ForObjectDetection.from_pretrained("ds4sd/docling-layout-heron")

    # Run the prediction pipeline
    inputs = image_processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = image_processor.post_process_object_detection(outputs,target_sizes=torch.tensor([image.size[::-1]]),threshold=0.6,)
    return results
    

def pdf_page_to_img(page):
    image = page.render(scale=4).to_pil()
    # image.save(f"output_{i:03d}.jpg")
    image = np.array(image)
    #image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

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


