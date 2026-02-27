from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

def analyze_cleanliness(image_path):
    image = Image.open(image_path).convert("RGB")

    # Step 1: Generate image caption
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Step 2: Rule-based hygiene scoring
    caption_lower = caption.lower()

    score = 80  # base score

    dirty_keywords = ["trash", "garbage", "dirty", "stain", "mess", "clutter"]
    clean_keywords = ["clean", "organized", "neat", "tidy"]

    for word in dirty_keywords:
        if word in caption_lower:
            score -= 15

    for word in clean_keywords:
        if word in caption_lower:
            score += 5

    score = max(0, min(100, score))

    return caption, score


if __name__ == "__main__":
    caption, score = analyze_cleanliness("restaurant.jpg")

    print("\nGenerated Caption:")
    print(caption)

    print("\nEstimated Cleanliness Score:", score, "/100")