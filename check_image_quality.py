import os
from PIL import Image
parent_path = os.getcwd()
directory = "Converted"
convert_path = os.path.join(parent_path,directory)
if os.path.exists(convert_path) == False:
    os.mkdir(convert_path)


im = Image.open('images/model04.jpeg')
im.thumbnail((512,512))
im.save('images/pexels1__.jpg',"PNG",dpi=(300,300))
