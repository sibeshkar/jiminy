import demo_pb2
from jiminy.vncdriver.screen import NumpyScreen

class Framebuffer(object):
    def __init__(self, self, width, height, server_pixel_format, name):
        self.width = width
        self.height = height
        self.name = name

        self.numpy_screen = screen.NumpyScreen(width, height)
        self.apply_format(server_pixel_format)

    def apply_format(self, server_pixel_format):
        self.bpp = server_pixel_format.get("bpp")
        self.depth =  server_pixel_format.get("depth")
        self.bigendian = server_pixel_format.get("bigendian")
        self.truecolor = server_pixel_format.get("truecolor")
        self.redmax = server_pixel_format.get("redmax")
        self.greenmax = server_pixel_format.get("greenmax")
        self.bluemax = server_pixel_format.get("bluemax")
        self.redshift = server_pixel_format.get("redshift")
        self.greenshift = server_pixel_format.get("greenshift") 
        self.blueshift = server_pixel_format.get("blueshift")

        self.bypp = self.bpp // 8
        shifts = [self.redshift, self.greenshift, self.blueshift]
        assert set(shifts) == set([0, 8, 16]), 'Surprising pixelformat: {}'.format(self.__dict__)
        #assert set(shifts) == set([10, 5, 0]), 'Surprising pixelformat: {}'.format(self.__dict__)
        # How to cycle pixels from images to get RGB
        self.color_cycle = np.argsort(shifts)

        self.numpy_screen.color_cycle = self.color_cycle

class FramebufferUpdate(object):
    def __init__(self, rectangles):
        self.rectangles = rectangles

class Rectangle(object):
    def __init__(self, x, y, width, height, encoding):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.encoding = encoding


