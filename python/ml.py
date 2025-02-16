import numpy as np


def imagexy_to_polar(x_pixel:float,y_pixel:float,x_0:float,rlim:float)-> np.ndarray:
    """transforms a coordinate of a point on an image (x,y=0,0 at top_left) to polar colatitude and longitude.
    x_pixel: x coordinate of the point
    y_pixel: y coordinate of the point
    x_0: x coordinate of the center of the image/ origin of the polar coordinate system.
    assumes the height=1/2 width of the image, and is calibrated to 90°<=colat<=270°
    """
    vec_p0 = np.array([x_0,0])
    vec_p = np.array([x_pixel,y_pixel])
    vec_pcal = vec_p - vec_p0
    pcal = np.linalg.norm(vec_pcal)
    colat = pcal/x_0 * rlim
    lon = np.degrees(np.arctan2(vec_pcal[1], vec_pcal[0])) + 270
    return np.array(colat,lon)
def imagexy_to_polar_arr(xy,x_0,rlim)->np.ndarray:
    """transforms a list of coordinates of points on an image (x,y=0,0 at top_left) to polar colatitude and longitude.
    xy: list of x,y coordinates of the points
    x_0: x coordinate of the center of the image/ origin of the polar coordinate system.
    assumes the height=1/2 width of the image, and is calibrated to 90°<=colat<=270°
    """
    return np.array([imagexy_to_polar(x,y,x_0,rlim) for x,y in xy])