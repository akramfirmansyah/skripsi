import cv2
from datetime import datetime
from picamera2 import Picamera2
import libcamera

filepath = "./images/" + datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".jpeg"

cam = Picamera2()
camera_config = cam.create_still_configuration(main={"size": (1920, 1080), "format":"RGB888"})
camera_config['transform'] = libcamera.Transform(hflip=1, vflip=1)
cam.configure(camera_config)

cam.start()

img = cam.capture_array()

isDone = cv2.imwrite(filepath, img)

if isDone:
    print(f"Capture Success in {filepath}!")
else:
    print("Capture Failed!")