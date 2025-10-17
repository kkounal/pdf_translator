from googletrans import Translator
translator = Translator()

def translate_text(sentances):
    for i,sentance in enumerate(sentances):
        while True:
            try:
                sentances[i] = translator.translate(sentance, src="en", dest="el").text
                break
            except Exception as e: 
                print("\033[93m Warning: exception at translation, probably a  timeout \033[0m")
                print(e)
    text = " ".join(sentances)
    return text


