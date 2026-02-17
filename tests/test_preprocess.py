from src.preprocess import resize_image

def test_resize():
    img = resize_image("sample.jpg")
    assert img.shape == (3,224,224)
