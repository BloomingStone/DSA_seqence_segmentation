from enum import StrEnum

class ImageType(StrEnum):
    Image = "image"
    Label = "label"
    FirstImage = "first_image"

    @classmethod
    def get_image_list(cls):
        return [cls.Image, cls.FirstImage]

    @classmethod
    def get_binary_image_list(cls):
        return [cls.Label]