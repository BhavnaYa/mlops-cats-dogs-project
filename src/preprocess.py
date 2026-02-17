from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def resize_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image)
