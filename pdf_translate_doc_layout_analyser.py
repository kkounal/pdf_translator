from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import torch
from PIL import Image

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

def results_interpreter(results):
    text_labels = set(['Caption','List-item','Text','Title','Senction-header', 'Table'])
    confidence_scores = []
    labels = []
    boxes = []
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = round(score.item(), 2)
            label = classes_map[label_id.item()]
            box = [round(i, 2) for i in box.tolist()]
            confidence_scores.append(score)
            labels.append(label)
            boxes.append(box)

    return confidence_scores, labels, boxes, text_labels

def detect_layout(img):
    image = Image.fromarray(img)
    image_processor = RTDetrImageProcessor.from_pretrained("ds4sd/docling-layout-heron")
    model = RTDetrV2ForObjectDetection.from_pretrained("ds4sd/docling-layout-heron")

    # Run the prediction pipeline
    inputs = image_processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = image_processor.post_process_object_detection(outputs,target_sizes=torch.tensor([image.size[::-1]]),threshold=0.6,)
    return results
    

"""
from ultralytics import YOLO

def results_interpreter(results):
    text_labels = set(['Caption','List-item','Text','Title','Senction-header', 'Table'])
    confidence_scores = []
    labels = []
    boxes = []
    
    boxes1 = results[0].boxes
    names = results[0].names
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    clsid = boxes.cls.cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        score = conf[i]
        class_id = int(clsid[i])
        class_name = names[class_id]
        confidence_scores.append(score)
        labels.append(class_name)
        boxes.append(x1,y1,x2,y2)

    return confidence_scores, labels, boxes, text_labels

def detect_layout(img):
    #yolo for image layout detection
    model_path = "yolov12l-doclaynet.pt"
    model = YOLO(model_path)
    #img = preprocessor.preprocess_image(img)
    results = model(img, device="cpu")  # Force CPU
    return results
"""







