from model import ImageCaptioning


class PythonPredictor:
    
    def __init__(self, config):
        """ load a model. """
        self.image_captioning = ImageCaptioning()

    def predict(self, payload):
        """ run a prediction. """
        image_url = payload["url"]
        output = self.image_captioning.run(image_url)
        return output