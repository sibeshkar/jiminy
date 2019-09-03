from jiminy.sdk.wrappers import Block

class ImageDataBlock(Block):
    def __init__(self, name=""):
        super(ImageDataBlock, self).__init__(name=name+"ImageDataBlock",
                input_dict={
                    "img" : [None, None, 3]
                    },
                output_dict={
                    "img" : [None, None, 3]
                    })

        def _forward(self, inputs):
            # perform pre-processing
            return inputs

if __name__ == "__main__":
    imb = ImageDataBlock()
