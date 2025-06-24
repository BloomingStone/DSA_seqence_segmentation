from enum import StrEnum

class ImageType(StrEnum):
    Image = "image"
    Label = "label"

    @classmethod
    def get_image_list(cls) -> list[str]:
        return [str(cls.Image)]

    @classmethod
    def get_binary_image_list(cls) -> list[str]:
        return [(cls.Label)]