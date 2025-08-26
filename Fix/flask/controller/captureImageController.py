from datetime import datetime
from constant.directory import images_dir

import cv2
import libcamera
from picamera2 import Picamera2
from flask import jsonify


def capture_image():
    """
    Captures an image using the camera and saves it to public/images directory.

    Returns:
        str: The filepath of the captured image if successful, None if failed.
    """
    try:
        filepath = (
            f"{images_dir}" + datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".jpeg"
        )

        cam = Picamera2()
        camera_config = cam.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )
        camera_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
        cam.configure(camera_config)

        cam.start()

        img = cam.capture_array()

        isDone = cv2.imwrite(filepath, img)

        cam.stop()

        if isDone:
            return filepath
        else:
            return jsonify({"status": "error", "message": "Failed to save image"}), 500

    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"Failed to capture image: {str(e)}"}
            ),
            500,
        )
