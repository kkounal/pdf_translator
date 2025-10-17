import re
import cv2
import numpy as np

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|ai|edu|co.uk|ru|info|biz|online)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = 255 - img
    img = cv2.merge([img,img,img])
    return img

def find_image_colors(img):
    pixel_vals = img.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
    k = 2
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    simg = segmented_data.reshape((img.shape))
    simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
    
    unique = np.unique(labels, return_counts=True, equal_nan=False)
    if unique[1][0] > unique[1][1]:
        index = np.where(labels == 0)[0]
        random_pixel_index = np.random.choice(index)
        random_pixel_index = np.unravel_index(random_pixel_index,(img.shape[0],img.shape[1]))
        bcolor = simg[random_pixel_index]

        index = np.where(labels == 1)[0]
        random_pixel_index = np.random.choice(index)
        random_pixel_index = np.unravel_index(random_pixel_index,(img.shape[0],img.shape[1]))
        fcolor = simg[random_pixel_index]
    else:
        index = np.where(labels == 1)[0]
        random_pixel_index = np.random.choice(index)
        random_pixel_index = np.unravel_index(random_pixel_index,(img.shape[0],img.shape[1]))
        bcolor = simg[random_pixel_index]

        index = np.where(labels == 0)[0]
        random_pixel_index = np.random.choice(index)
        random_pixel_index = np.unravel_index(random_pixel_index,(img.shape[0],img.shape[1]))
        fcolor = simg[random_pixel_index]
    return simg,tuple(bcolor),tuple(fcolor)



