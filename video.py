import cv2

class VideoCapture:
    def __init__(self, gstreamer_pipeline = 0, shape = None):
        # if not type(gstreamer_pipeline) == str or gstreamer_pipeline!=0:
        #     raise TypeError('Gstreamer pipeline should be string or integer=0.')
        self.pipeline = gstreamer_pipeline
        self.cap = cv2.VideoCapture(self.pipeline)
        self.resize_flag = False
        if shape is not None:
            self.resize_flag = True
            self.shape = (shape[1], shape[0]) # Change order of shape because opencv style
        else:
            self.shape = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_count = 0

    def read(self):
        grabbed, frame = self.cap.read()
        if not grabbed:
            raise BufferError('No frame in buffer')
        # If not grabbed send message to main thread
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.resize_flag:
            frame = cv2.resize(frame, self.shape)
        self.frame_count += 1
        return frame

    def rgb2bgr(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def close(self):
        self.cap.release()