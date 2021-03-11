"""
    Computer vision and statistical utilities
    Copyright (C) 2021 Amélia O. F. da S.
    <a.mellifluous.one@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import math
import cv2
def smooth(array:np.ndarray,newarray:np.ndarray,factor:float)->np.ndarray:
    """
    Effectively smoothes variations in "array" when updating it with "newarray"
    The result is an exponential smoothing.
    For a constant "newarray", for example:
    If the smoothing factor is 80%, it'll take 21 cycles for "array" to
    be 99% composed by "newarray";
    If the smoothing factor is 30%, it'll take 4 cycles for "array" to
    be 99% composed by "newarray";
    If the smotthing factor is 0, array = newarray
    """
    return (factor*array.astype(np.float32)+(1-factor)*newarray.astype(np.float32)).astype(array.dtype)
def locmin(img:np.array,x:int,y:int,w:int,pimg:np.array,w2:int)->tuple:
    """
    Effectively tracks high-contrast patterns.
    In practive, finds the area within the wXw area around (x,y)
    with the biggest cross-correlation to the original image
    Returns ((new x, new y), new pimg, min_difference)
    Can take None as pimg (for, e.g. resetting or the first iteration).
    """
    #If we're on an edge
    if x<w or img.shape[1]-x<w or y<w or img.shape[0]-y<w :
        #Halt or else we'll end up crashing the script
        return (np.array([0,0]),pimg,0)
    #If there's no image to track
    if isinstance(pimg,type(None)):
        #Return one centered on (x,y)
        return (np.array([x,y]),img[y-w2:y+w2,x-w2:x+w2],0)
    #Separate the area around the point
    area = img[y-w:y+w,x-w:x+w]
    area = cv2.equalizeHist(area).astype(np.float32)
    #Calculate the modulus of our kernel area
    km = pimg.sum()
    #Calculate the vector moduli of each area
    moduli = cv2.filter2D(area,-1,np.ones(area.shape))
    #Make the vector product
    #filter2D reflects the kernel through its centre, so we revert pimg on both axes
    scores = np.abs(cv2.filter2D(area,-1,pimg[::-1,::-1].astype(np.float32),borderType=cv2.BORDER_ISOLATED))
    #Divide the product by the product of the moduli at each point
    scores = (scores/km)/moduli
    #Limit it to the area where our data is actually meaningful
    scores = scores[w2:-w2,w2:-w2]
    #Find the minimum
    om = np.unravel_index(np.argmin(scores),scores.shape)
    m = (om[0]+w2,om[1]+w2)
    return (np.array([x+(int(m[1]-w)),y+(int(m[0]-w))]),smooth(pimg,area[m[1]-w2:m[1]+w2,m[0]-w2:m[0]+w2],0.9),scores[om])
def point_is_outlier(point:np.array,lastn:np.array,var_threshold:float,length:int=-1,output_current_ratio:bool=False)->tuple:
    """
    Determines whether a data point is an outlier through variance and adds it to fixed-size set.
    For the first point, the function expects lastn=None and length=desired set size.
    For the others, it expects lastn to be the set returned by the last iteration
    If (point-mean)**2/(len(set)*variance(set))-1>var_threshold, returns (True,newset)
    otherwise, returns (False,newset)
    """
    if lastn is None:
        if output_current_ratio:
            return (False,np.array([point]),0)
        return (False,np.array([point]))
    else:
        if(len(lastn)<length):
            lastn=np.insert(lastn,0,np.zeros(point.shape),0)
            if output_current_ratio:
                return (False,lastn,0)
            else:
                return (False,lastn)
        #Shift the current values left
        lastn[0:-1] = lastn[1:]
        #Put our last point in the rightmost position
        lastn[-1] = point
        #Calculate the variance
        var = lastn.var()
        if var == 0:
            #To avoid NaNs
            var = 1e-10
        #Mean
        mean = lastn.mean(0)
        ratio = ((point-mean)**2).sum()/(len(lastn)*var)
        if output_current_ratio:
            return (ratio>var_threshold,lastn,ratio)
        else:
            return (ratio>var_threshold,lastn)
def real_xyz_from_screen_xy(current_vector2d_screen:np.ndarray,screen_centre:np.ndarray,
                            screen_length_at_reference_z:float,reference_z:float,
                            real_length:float)->np.ndarray:
    """
    Uses triangle similarities to calculate object coordinates with the camera as (0,0,0),
    Y being "up" on the camera image, X being "right" and Z being "into"
    from reference measurements at a known distance
        current_vector2d_screen = [[x0,y0],[x1,y1]]
        screen_centre = [x,y]
    """
    ret = np.zeros(3)
    #Z coordinate
    #Going further in the Z direction scales our XY plane, so we calculate that scaling factor
    scaling_factor_2d = screen_length_at_reference_z/np.linalg.norm(current_vector2d_screen[0]-current_vector2d_screen[1])
    if scaling_factor_2d==math.inf:
        scaling_factor_2d=0
    #And multiply it by our reference Z
    ret[2] = reference_z*scaling_factor_2d
    #X and Y coordinates
    vector2d_centre = (current_vector2d_screen[0]+current_vector2d_screen[1])/2
    distance_2d_from_centre = screen_centre-vector2d_centre
    #Using the same scaling factor, we can estimate what would be the size, in pixels, of our
    #position in that XY plane at the reference distance. We just need to use the scaling factor
    #in reverse (dividing by it instead of multiplying)
    projected_distance_at_reference = distance_2d_from_centre/scaling_factor_2d if scaling_factor_2d!=0 else 0
    #That distance is proportional to the real XY position through real_length/screen_length_at_reference_z
    ret[:2] = real_length * projected_distance_at_reference/screen_length_at_reference_z
    return ret
def real_xyz_from_screen_xy_mtx(current_vector2d_screen:np.ndarray,screen_centre:np.ndarray,
                                real_length:float,inverse_intrinsic_matrix:np.ndarray)->np.ndarray:
    """
    Uses a calibrated camera matrix to calculate the real-life coordinates
    for the object with the camera as (0,0,0), Y being "up" on the camera image,
    X being "right" and Z being "into"
        current_vector2d_screen = [[x0,y0],[x1,y1]]
        screen_centre = [x,y]
    """
    #If both trackers are at the same point, we can't determine the position
    if np.all(current_vector2d_screen[0]-current_vector2d_screen[1]==np.array([0,0])):
        return np.array([0,0,0])
    #We'll first transform the screen coordinates into 
    #direction vectors coming from the camera origin
    a  = np.dot(inverse_intrinsic_matrix,np.insert(current_vector2d_screen[0],2,1))
    na = np.linalg.norm(a)
    a  = a/na
    b  = np.dot(inverse_intrinsic_matrix,np.insert(current_vector2d_screen[1],2,1))
    nb = np.linalg.norm(b)
    b  = b/nb
    #cos(aOb)=ab/|a||b|
    #|a|=|b|=1, so cos(aOb)=ab
    cos = np.dot(a,b)
    theta = math.acos(cos)
    #tan(alpha)= opposite side/adjacent side
    #opposite side = real length/2, adjacent side = z distance, alpha=theta/2
    #z distance = real length/2*tan(theta/2)
    z=0
    z = real_length/(2*math.tan(theta/2))

    #Now for calculating the XY position on the plane,
    #we calculate the ray going to the centre of the line segment
    #we're tracking
    centre = (current_vector2d_screen[0]+current_vector2d_screen[1])/2
    d = np.dot(inverse_intrinsic_matrix,np.insert(centre,2,1))
    #We need to scale this direction vector so its z component is at our depth
    scaling = z/d[2]
    d = d*scaling
    #The Y direction is flipped
    d[:2]=-d[:2]
    #This is our exact position
    return d
def project_image_at_xyz(img:np.ndarray,xyz:np.ndarray,target_window_shape:tuple,
                            reference_z:float,fixedscale:float,fixedtranslation:tuple,
                            screen_pixel_density:float):
    """
    Deforms an image so it looks static for a viewer at the real position XYZ
    """
    #First triangle similarity: XY planes for each z
    plane_scaling = xyz[2]/reference_z
    #Translation
    #We know it is at position XY in the real world, so we have to move it by
    #XY*screen_pixel_density at the reference for it to stay aligned to us
    screenXY = (xyz[:2]*screen_pixel_density)
    screenXY[1] = - screenXY[1]
    #Calculating the affine transform
    #|S0X|
    #|0SY|
    #This scale is simply our plane scaling times a fixed scale we can use to make the image bigger
    scale = fixedscale*plane_scaling
    transform = np.array([
        [scale,0    ,fixedtranslation[0]+(screenXY[0]+target_window_shape[0]/2-scale*img.shape[1]/2)],
        [0    ,scale,fixedtranslation[1]+(screenXY[1]+target_window_shape[1]/2-scale*img.shape[0]/2)]
    ])
    #Transforming the image
    return cv2.warpAffine(img,transform,target_window_shape)