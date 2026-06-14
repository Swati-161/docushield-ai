from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')

dummy = np.ones((100, 400, 3), dtype=np.uint8) * 255
result = ocr.predict(dummy)

print("PaddleOCR models downloaded and cached successfully!")