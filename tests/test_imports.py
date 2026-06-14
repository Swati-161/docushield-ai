import cv2
import torch
import torchvision
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from fastapi import FastAPI
import spacy

print("cv2 version:        ", cv2.__version__)
print("torch version:      ", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("spacy version:      ", spacy.__version__)
print("fastapi imported    : OK")
print("paddleocr imported  : OK")
print("pdf2image imported  : OK")
print()
print("All imports successful!")