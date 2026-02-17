from src.preprocess import resize_image
from PIL import Image
import os

def test_preprocess_shape():
    # Create a dummy image
    img = Image.new("RGB", (300, 300), color="red")
    img.save("temp_test_image.png")

    # Run preprocess
    tensor = resize_image("temp_test_image.png")

    # Check shape
    assert tensor.shape == (3, 224, 224)

    # Clean up
    os.remove("temp_test_image.png")
