# Include as many packages as you'd like here
import numpy as np
from PIL import Image
import skimage.transform
import cv2
import matplotlib.pyplot as plt

savedir = "images/"


def save_fig_as_png(figtitle):
    '''
    
    Saves the current figure into the output folder specificed by the variable "savedir".
    Note: depending on the OS you might change the backslashes / to \.
   
    The figtitle should not contain the ".png".
    
    This helper function should be easy to use and should help you create/save the figures 
    needed for the report.
    
    Hint: The plt.gcf() might come in handy
    Hint 2: read about this to crop white borders
    https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content  
    
    Args:
        figtile: filename without the ending ".png"
        
    '''
    
    fig = plt.gcf()
    path = savedir+figtitle+".png"
    plt.savefig(path,bbox_inches='tight', pad_inches=0)
    plt.show()

    return 0


def load_image(path):
    """
    TODO: IMPLEMENT ME
    
    Loads an image specified by path.
    
    Specifications on the image:
        1. Image should be returned with type float32 
        2. Image should be scaled between 0 and 1
        3. If the image has a transprancy channel, the output is a 4-channel array
            a) You can test with the image "dog.png" which has an alpha channel

    Args:
        path: path to the file
    Returns:
        output (np.ndarray): The northwesten image as an RGB image (or RGB alpha if 4 channel image)
    """
    
    img=np.asarray(Image.open(path))
    img=img.astype(np.float32)
    img=(img-np.min(img))/(np.max(img)-np.min(img))
    
    return img
    

def crop_chicago_from_northwestern(img):
    """
    TODO: IMPLEMENT ME
    
    Crop a region-of-interest (ROI) from the big northwestern image that shows only Chicago
    
    The image size should be (250, 1000) and the the output should be an RGB numpy array
    
    Args:
        input (nd.array): The image of Northwestern and Chicago
    Returns:
        output (np.ndarray): The skyline of chicago with size (250,1000,3)
    """
    
    center_x=img.shape[1]//2 # It gives the central value in abscisa
    
    chicago=img[70:320,center_x-500:center_x+500,:] # We select the interest area (first 250 pixels in OY and central values on OX)
    return chicago
    
def downsample_by_scale_factor(img,scale_factor):
    """
    TODO: IMPLEMENT ME
    
    Downsample the input image img by a scaling factr
    
    E.g. with scale_factor = 2 and img.shape = (200,400)
    
    you would expect the output to be (100,200)
    
    You can use external packages for downsampling. Just look 
    for the right package

    Args:
        input (nd.array): The image of Northwestern and Chicago
    Returns:
        output (np.ndarray): The third dimension shouldn't change, only the first 2 dimensions.
    """
    
    chicago_downsample=skimage.transform.resize(img, (img.shape[0]//scale_factor, img.shape[1]//scale_factor),anti_aliasing=True)
    chicago_downsample=np.asarray(chicago_downsample)
    chicago_downsample=chicago_downsample.astype(np.float32)
    
    return chicago_downsample
    



def convert_rgb2gray(rgb):
    """
    TODO: IMPLEMENT ME
    
    rgb2gray converts RGB values to grayscale values by forming a weighted
    sum of the R, G, and B components:

    0.2989 * R + 0.5870 * G + 0.1140 * B 
    
    
    These values come from the BT.601 standard for use in colour video encoding,
    where they are used to compute luminance from an RGB-signal.
    
    Find more information here:
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf

    Args:
        input (nd.array): 3-dimensional RGB where third dimension is ordered as RGB
    Returns:
        output (np.ndarray): Gray scale image of RGB weighted by weighting function from above
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray

def plot_chicago_skyline(img):
    """
    TODO: IMPLEMENT ME
    
    This is a simple exercise to learn how to use subplot.
    
    Goal of is to show a 2x2 subplot that shows the Chicagskyline for 
    4 different downsampling factors: 1,2,4,8
    
    Use plt.subplot to create subfigures
    
    You should give a title of the compelte image (use plt.suptitle)
    and each subfigure should have a corresponding title as well.

    Args:
        input (nd.array): 2-dimensional gray scale image
    Returns:
        
    """
    img2=downsample_by_scale_factor(img,2)
    img4=downsample_by_scale_factor(img,4)
    img8=downsample_by_scale_factor(img,8)
    
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15,5))
    fig.suptitle("Chicagskyline")
    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Downsampling factor 1")

    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title("Downsampling factor 2")

    ax[2].imshow(img4, cmap='gray')
    ax[2].set_title("Downsampling factor 4")

    ax[3].imshow(img8, cmap='gray')
    ax[3].set_title("Downsampling factor 8")

    plt.tight_layout()
    plt.show()

def rescale(img,scale):
    """
    TODO: IMPLEMENT ME
    
    Implement a function that scales an image according to the scale factor
    defined by scale
    
    If you're using the rescale function from scikit-learn make sure
    that it is not rescaling the 3rd dimension. 
    
    Look at the output of the image and see if looks like expected,
    if not, come up with a solution that solves this problem.

    """    
    img_new=skimage.transform.rescale(img, scale, anti_aliasing=False,multichannel=True)
    img_new=np.asarray(img_new)
    img_new=img_new.astype(np.float32)
    
    return img_new

def pad_image(img,pad_size):
    """
    TODO: IMPLEMENT ME
    
    Takes an image and pads it symmetrically at all borders with
    pad_size as the size of the padding

    Args:
        img (np.ndarray): image to be padded
    Returns:
        output (np.ndarray): padded image
    """    

    top, bottom = pad_size//2, pad_size-(pad_size//2)
    left, right = pad_size//2, pad_size-(pad_size//2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    return new_img

def add_alpha_channel(img):
    """
    TODO: IMPLEMENT ME
    
    Takes an image with 3 channels and adds an alpha channel (4th channel) to it
    Alpha channel should be initialize so that it is NOT transparent
    
    Think about what value this should be!
    
    Args:
        img (np.ndarray): rgb imagew without alpha channel
    Returns:
        output (np.ndarray): rgb+depth image
    """    
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)*0.8 #creating a dummy alpha channel image.
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    return img_BGRA

def overlay_two_images_of_same_size(img1,img2):
    """
    TODO: IMPLEMENT ME

    This is a helper function that can be used to implement
    the function "overlay_two_images"
    
    This function takes 2 image of the same input size
    and adds them together via simple superposition.
    
    WARNING: You have to account for the alpha-channel of img2
    to correct for nice superposition of both images
    
    Hint: https://en.wikipedia.org/wiki/Alpha_compositing
    
    Args:
        img1 (nd.array): The image of the background
        img2 (nd.array): The image to be overlayed (e.g. the dog) that has the shape of img1. Img2 should have an alpha channel that has non-zero entries
        location (nd.array): x,y coordinates of the top-left of the image you want to overlay
    Returns:
        output (np.ndarray): An image of the same size   
    
    
    mask = img1[:,:,3]
    
    img1=(img1-np.min(img1))/(np.max(img1)-np.min(img1))
    img2=(img2-np.min(img2))/(np.max(img2)-np.min(img2))
    
    b=np.zeros((img1.shape[0],img1.shape[1]))
    g=np.zeros((img1.shape[0],img1.shape[1]))
    r=np.zeros((img1.shape[0],img1.shape[1]))
    #new_img=np.zeros((img1.shape[0],img1.shape[1],3))
    
    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            b[i,j]=mask[i,j]*img2[i,j,0] + img1[i,j,0]
            g[i,j]=mask[i,j]*img2[i,j,1] + img1[i,j,1]
            r[i,j]=mask[i,j]*img2[i,j,2] + img1[i,j,2]
    
    new_img= np.dstack((b,g,r))
    
    return new_img
    """ 
    img_new=img1
    for i in range(3):
        img_new[:,:,i] = img1[:,:,i]*(1-img2[:,:,3]) + img2[:,:,i]*img2[:,:,3]

    return img_new
    
def overlay_two_images(img1,img2,location):
    """
    TODO: IMPLEMENT ME

    Overlays a background image (img1) with a forgeground image (img1)
    Location defines the tope-left location where img2 is placed ontop of
    the background image1
    
    NOTE: img2 can be a large image and its boundaries could go over
    the image boundaries of the background img1.
    
    You'll have to crop img2 accordingly to fit into img1 and to avoid
    any numpy errors (out-of-bound errors)
    
    Hint: https://en.wikipedia.org/wiki/Alpha_compositing
    
    Args:
        img1 (nd.array): The image of the background
        img2 (nd.array): The image to be overlayed (e.g. the dog)
        location (nd.array): x,y coordinates of the top-left of the image you want to overlay
    Returns:
        output (np.ndarray): An image of size img1.shape that is overlayed with img2
    """
    
    crop=img2[:780,:,:] # Cutting dog paws
    
    # Locations and length of the images
    lx=location[0]
    ly=location[1]
    imgx=crop.shape[0]
    imgy=crop.shape[1]
    
    img_new=img1
    
    for i in range(3):
        img_new[lx:lx+imgx,ly:ly+imgy,i] = img1[lx:lx+imgx,ly:ly+imgy,i]*(1-crop[:,:,3]) + crop[:,:,i]*crop[:,:,3]

    return img_new

