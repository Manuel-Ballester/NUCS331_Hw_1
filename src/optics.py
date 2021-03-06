import numpy as np
import scipy

def create_radial_distance_map(N_img):
    """  
    This should create 2-Dimensional matrix which calculates
    the radial distance from the center. I.e. the value 
    in the center should be 0 and then the values increase
    radial symmetric around the center.
    
    Helpful function are np.linspace, np.meshgrid
    and the np.sqrt function
    
    You can use Pythagorean theorem to calculat the radial distances.
    
    Args:
        N_img: Number of pixels of the distance map
    Return:
        R (float): Radial distance map as 2D array
    """
    array=np.zeros((N_img,N_img))
    center=N_img//2 # If N_img even, it takes the previous value
    
    for i in range(N_img):
        for j in range(N_img):
            array[i,j]=np.sqrt(np.abs(i-center)**2+np.abs(j-center)**2)

    return array

def gaussian_psf(R,sigma):
    """  
    Gaussian PSFs creates a 2D matrix which follows
    a Gaussian Point-spread-function.
    
    Find more information on what a PSF is here:
    https://en.wikipedia.org/wiki/Point_spread_function
    
    Calculating this is quite simple and you don't
    require any additional packages.
    
    Simply apply the gaussian function 
    (https://en.wikipedia.org/wiki/Gaussian_function)
    to your radial distance map.
    
    The only parameter you'll need is sigma. Simply
    set the mean to 0.
    
    Args:
        R: radial distance map
        sigma: standard deviation of the gaussian PSF
    Return:
        PSF (float): A 2D-array with the gaussian PSF
    """
    gauss=1/(sigma*np.sqrt(2*np.pi))*np.exp(-R**2/(2*sigma**2))
    return gauss/np.sum(gauss)



def calc_angular_field_of_view(sensor_size_mm,focal_length):
    """
    Calcualte Angle of View
    
    According to: https://shuttermuse.com/calculate-field-of-view-camera-lens/

    Args:
        sensor_size_mm (float): sensor size of camera
        focal_length (float): focal length of camera
    Returns:
        angle of view of specific camera
    """
    angle=2*np.arctan(sensor_size_mm/(2*focal_length))* (180/np.pi)
    return angle


def calc_field_of_view(sensor_size_mm,o_obj,focal_length):
    """
    Calculate linear field of view at specific distance away from the lens
   
    
    You have to transform the equation given in https://en.wikipedia.org/wiki/Magnification#Photography

    Args:
        sensor_size_mm (float): sensor size of camera
        o_obj (float): distance where object is located
        focal_length (float): focal length of objetive lens
    Returns:
        linear of view of specific camera for both dimensions in mm
    """
    angle=calc_angular_field_of_view(sensor_size_mm,focal_length)
    FoV=2*(np.tan(angle/2)*o_obj)
    
    return FoV


def calc_blur_radius(f,D,o_foc,o_obj):
    """
    Calcualte the blur radius according to
    lecture 1 - image formation

    Args:
        sensor_size_mm (float): sensor size of camera
        o_obj (float): distance where object is located
        focal_length (float): focal length of objetive lens
    Returns:
        angle of view of specific camera
    """
    
    b=D*abs((f*(o_foc-o_obj))/(o_obj*(o_foc-f)))
    return b
    


def crop_background_image_sensor_ratio(sensor_size_mm,img):
    """"
    
    This functions crops an image of arbitrary to size to a specific aspect ratio defined 
    by the sensor size of the camera
    
    1. Calculate aspect ratio of sensor: e.g., 24/36 = 2/3
    2. Calculate aspect ratio r of input image:
        Two cases: r > 2/3 or r<2/3
    Depending on those cases, you have to crop either width or height (think about which one)
    3. Depending on result of (2), calculate how much you have to crop
    4. Crop only either width or height dimension, depending on what you've calculated in 3
    5. You're done. I would use only numpy for this. Nothing else is needed

    
    Input:
    sensor_size (float) : array containing the height and width of the image sensor which defines the aspect ratio
    img (int,float) : the image that should be cropped to the specific aspect ratio
    Output:
    resized_img (int,loat): The cropped image that now has the correct aspect ratio
    """
    
    # Sensor and image ratios
    sensor_r=sensor_size_mm[0]/sensor_size_mm[1]
    img_r=img.shape[0]/img.shape[1]
    
    if sensor_r<=img_r:
        width=int(img.shape[0]/sensor_r)
        new_img=img[:,:width,:]
    else:
        height=int(sensor_r*img.shape[1])
        new_img=img[:height,:,:]
        
    return new_img

def convolve_image(img,PSF):
    """
    
    Convolve_image blurs the image with a PSF. Because we've defined very large blur 
    kernels (100x100 instead of small ones like 5x5) these Convolutions can take
    very long, especially when performed in spatial domain.
    
    The trick is to perform this convolution in frequency domain using the
    convolution theorem: https://en.wikipedia.org/wiki/Convolution_theorem
    
    Luckily, there are already python package which perform convolution
    in the frequency domain. 
    
    You can look e.g. at scipy.signal.fftconvolve for such a method that you
    can use in this functon.
    
    Input:
    img (float) : RGB array of the image to be blurred
    psf (float) : the point spread function we are blurring the image with
    Output:
    img_filtered (foat): The filtered image of same size as img
    """
    
    blurred=img
    
    for i in range(3):
        blurred[:,:,i]= scipy.signal.fftconvolve(img[:,:,i], PSF,mode="same")
        
    return blurred
