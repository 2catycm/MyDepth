import numpy as np
import cv2
from PIL import Image
def DepthMapPseudoColorize(depth_map,output_path, max_depth = None, min_depth = None):
    # Load 16 units depth_map(element ranges from 0~65535),
    # Convert it to 8 units (element ranges frome 0~255) and pseudo colorize
    if isinstance(depth_map, np.ndarray):
        uint16_img =  depth_map
    # Indicate the argument load mode -1, otherwise loading default 8 units
    if isinstance(depth_map, str):
        uint16_img = cv2.imread(depth_map, -1)
    if None == max_depth:
        max_depth = uint16_img.max()
    if None == min_depth:
        min_depth = uint16_img.min()

    uint16_img -= min_depth
    uint16_img = uint16_img / (max_depth - min_depth)
    uint16_img *= 255

    # cv2.COLORMAP_JET, blue represents a higher depth value, and red represents a lower depth value
    # The value of alpha in the cv.convertScaleAbs() function is related to the effective distance in the depth map. If like me, all the depth values
    # in the default depth map are within the effective distance, and the 16-bit depth has been manually converted to 8-bit depth. , then alpha can be set to 1.
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(uint16_img,alpha=1),cv2.COLORMAP_JET)
    #convert to mat png
    im=Image.fromarray(im_color)
    im.save(output_path)