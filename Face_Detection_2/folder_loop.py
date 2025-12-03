import os
import json
import numpy as np

for filename in os.listdir("Photos"):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    name = os.path.splitext(filename)[0]
    img_path = os.path.join("Photos", filename)

    print(img_path)

