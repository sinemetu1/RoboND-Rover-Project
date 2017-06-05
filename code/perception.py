import numpy as np
import cv2

## Identify pixels above the threshold
## Threshold of RGB > 160 does a nice job of identifying ground pixels only
#def color_thresh(img, rgb_thresh=(160, 160, 160)):
    ## Create an array of zeros same xy size as img, but single channel
    #color_select = np.zeros_like(img[:,:,0])
    ## Require that each pixel be above all three threshold values in RGB
    ## above_thresh will now contain a boolean array with "True"
    ## where threshold was met
    #above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                #& (img[:,:,1] > rgb_thresh[1]) \
                #& (img[:,:,2] > rgb_thresh[2])
    ## Index the array of zeros with the boolean array and set to 1
    #color_select[above_thresh] = 1
    ## Return the binary image
    #return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


def get_destination(image, dst_size, bottom_offset):
    return np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])


# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_min=(160, 160, 160), rgb_max=(256, 256, 256)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    meets_thresh = (img[:,:,0] > rgb_min[0]) \
                & (img[:,:,1] > rgb_min[1]) \
                & (img[:,:,2] > rgb_min[2]) \
                & (img[:,:,0] < rgb_max[0]) \
                & (img[:,:,1] < rgb_max[1]) \
                & (img[:,:,2] < rgb_max[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[meets_thresh] = 1
    # Return the binary image
    return color_select


def rocks_thresh(img):
    return color_thresh(img, rgb_min=(110, 110, 5), rgb_max=(255, 255, 90))


def obstacles_thresh(img):
    return color_thresh(img, rgb_min=(0, 0, 0), rgb_max=(20, 160, 160))


def within_range(v):
  return (v >= 0.0 and v <= 1.0) or (v >= 359.0 and v <= 360.0)


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    xpos, ypos = Rover.pos
    yaw = Rover.yaw
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = get_destination(Rover.img, dst_size, bottom_offset)
    warped = perspect_transform(Rover.img, source, destination)

    Rover.dists = {}
    Rover.angles = {}

    i = 2
    for name, m in (('nav', color_thresh), ('rocks', rocks_thresh), ('obstacles', obstacles_thresh)):
        threshed = m(warped)
        Rover.vision_image[:,:,i] = threshed * 255
        xpix, ypix = rover_coords(threshed)
        scale = 20
        x_world, y_world = pix_to_world(xpix, ypix, xpos, 
                                        ypos, yaw, 
                                        Rover.worldmap.shape[0], scale)
        if within_range(Rover.pitch) and within_range(Rover.roll):
            Rover.worldmap[y_world, x_world, i] += 1

        dist, angles = to_polar_coords(xpix, ypix)
        Rover.dists[name] = dist
        Rover.angles[name] = angles
        if name == 'nav':
            Rover.nav_dists = dist
            Rover.nav_angles = angles

        i =  i - 1
    return Rover
