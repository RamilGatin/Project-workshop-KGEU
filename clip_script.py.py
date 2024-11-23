# clip_script.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests

# Загрузка модели CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Функция для загрузки изображения
def load_image(image_path):
    return Image.open(image_path)


# Функция для сравнения изображения с текстовыми описаниями
def compare_image_with_texts(image, texts):
    # Преобразование изображения и текстов для модели CLIP
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # нормализуем вероятности
    return probs


# Загрузка изображения (вам нужно будет вручную передать путь к файлу)
image_path = "path/to/your/image.jpg"  # Укажите путь к своему изображению
image = load_image(image_path)

# Список текстовых описаний
texts = [
    "a dog running in the park",
    "a beautiful sunset over the mountains",
    "a person riding a bike",
    "a cat sitting on the windowsill",
    "a car driving through the city",
]

# Сравнение изображения с текстами
probs = compare_image_with_texts(image, texts)

# Печать результатов
for i, text in enumerate(texts):
    print(f"'{text}': {probs[0, i].item()*100:.2f}% similarity")

# Определение наилучшего описания
best_match_idx = torch.argmax(probs).item()
print(f"\nThe best matching description is: '{texts[best_match_idx]}'")
