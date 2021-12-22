import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Tiff:
    def __init__(self, path):
        self.img = Image.open(path)
        self.mask = None
        self.nb_roi = None
        self.conncomp = None
        self.T = self.get_t()
        self.Z = self.get_z()
        self.C = self.get_c()

    def to_numpy(self):
        return np.asarray(self.img)

    def get_imagej_metadata(self, formatted=True):
        ls = [v for v in self.img.tag.values()]
        for element in ls:
            if type(element) is not tuple:
                continue
            if len(element) == 0:
                continue
            if type(element[0]) is not str:
                continue
            if "ImageJ" in element[0]:
                if formatted:
                    return self.format_imagej_metadata(element[0])
                return element[0]

    def format_imagej_metadata(self, metadata):
        ret = {
            name: value
            for (name, value) in [
                tuple(x.split("=")) for x in metadata.split("\n") if "=" in x
            ]
        }
        return ret

    def get_t(self):
        dict_metadata = self.get_imagej_metadata(formatted=True)
        return int(dict_metadata.get("frames", 1))

    def get_z(self):
        dict_metadata = self.get_imagej_metadata(formatted=True)
        return int(dict_metadata.get("slices", 1))

    def get_c(self):
        dict_metadata = self.get_imagej_metadata(formatted=True)
        return int(dict_metadata.get("channels", 1))

    def seek_image(self, t=0, z=0, c=0):
        # if stuff for later
        self.img.seek(t * self.Z * self.C + z * self.C + c)
        return self

    def seek_z_only(self, z=0):
        self.img.seek(z)
        return self

    def region(self, origine, size, **kwargs):
        self.seek_image(**kwargs)
        if isinstance(size, int):
            size = (size, size)

        return self.to_numpy()[origine[0]: origine[0] + size[0],
                               origine[1]: origine[1] + size[1]]

    def get_image_as_array(self, **kwargs):
        self.seek_image(**kwargs)
        return np.asarray(self.img)

    def show(
        self, t=None, z=None, c=None
    ):  # take the seek, make the size bigger,
        # handle a list of seeks in subplot
        plt.figure(figsize=(10, 10))
        self.seek_image(t=t, z=z, c=c)
        plt.imshow(self.img)
        plt.show()
