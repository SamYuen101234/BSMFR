import pyrealsense2 as rs
import numpy as np
import cv2

# using Intel realsense depth camera to achieve anti-spoofing

class DC:
    def __init__(self, IMG_SIZE):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        self.config.enable_stream(rs.stream.depth, IMG_SIZE[0], IMG_SIZE[1], rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, IMG_SIZE[0], IMG_SIZE[1], rs.format.bgr8, 30)

    def start(self):
        self.pipeline.start(self.config)
    
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # return numpy depth and color img
        return {'depth_image': depth_image, 'color_image': color_image}

    def get_depth_colormap(self, depth_image):
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    def check_spoofing(self):
        pass

if __name__ == "__main__":
    IMG_SIZE = 640,480
    DEPTH_CAM = DC(IMG_SIZE)
    DEPTH_CAM.start()
    try:
        while True:
            images = DEPTH_CAM.get_frame()
            depth_colormap = DEPTH_CAM.get_depth_colormap(images['depth_image'])
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = images['color_image'].shape

            if depth_colormap_dim != color_colormap_dim:    # if the depth image dim != color image dim
                resized_color_image = cv2.resize(images['color_image'], dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((images['color_image'], depth_colormap))
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)
    finally:
        # Stop streaming
        DEPTH_CAM.pipeline.stop()
