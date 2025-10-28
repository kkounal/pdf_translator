from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def create_text_image(text,width,height,cfill = (255,255,255),cfont=(0,0,0),font_path = "ArialGreekRegular.ttf"):
    im =  Image.new(mode="RGB", size=(width, height))
    draw = ImageDraw.Draw(im, "RGB")
    text_width = width * 0.8
    text_max_height = height * 0.8
    size = 96
    while size > 1:
        font = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.BASIC)
        lines = []
        line = ""
        for word in text.split():
            proposed_line = line
            if line:
                proposed_line += " "
            proposed_line += word
            if font.getlength(proposed_line) <= text_width:
                line = proposed_line
            else:
                # If this word was added, the line would be too long
                # Start a new line instead
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        text = "\n".join(lines)

        x1, y1, x2, y2 = draw.multiline_textbbox((0, 0), text, font, stroke_width=0.7)
        w, h = x2 - x1, y2 - y1

        '''
        temp = im.copy()
        draw_t = ImageDraw.Draw(temp, "RGB")
        draw_t.rectangle([(0, 0), (width, height)], fill = (255,255,255))
        draw_t.multiline_text((width / 2 - w / 2 - x1, height / 2 - h / 2 - y1), text, font=font, align="left", stroke_width=0.7, fill=(0,0,0))
        gray = cv2.cvtColor(np.array(temp), cv2.COLOR_BGR2GRAY)
        cv2_t = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
        rowSums = np.sum(cv2_t[[0,-1], :]-255, axis=1)
        threshold = cv2_t.shape[0] * 255
        if rowSums[0] > threshold or rowSums[-1] > threshold:
            break
        '''
        if int(h) <= height:
            break
        else:
            # The text did not fit comfortably into the image
            # Try again at a smaller font size
            size -= 1
    
    draw.rectangle([(0, 0), (width, height)], fill = cfill)
    draw.multiline_text((width / 2 - w / 2 - x1, height / 2 - h / 2 - y1), text, font=font, align="left", stroke_width=0.7, fill=cfont)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


