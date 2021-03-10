# Eye-tracking based experiments
By Amélia O. F. da S.

---

## Purpose

These experiments were all done for educational and entertainement purposes. Hopefully it'll help others understand some techniques better as much as making them helped me.

## Experiments

### Position estimation and projecting a static 3d image
* Run `positionestimator.py`
* Estimates a person's position relative to the screen and uses it to project an image that looks static in 3d space to them.
* How to use:
    * Open `positionestimator.py`, click on your eyes on the "Tracker" window to start tracking them, stay at a known distance from the screen and register the number on the top left corner of the screen. Measure your pupil separation in the same units as your reference distance. Measure the width of the "3d Static Image" window in the same units as you measured the previous distances. Edit the constants at the start of the file to match your measurements.
    * Now if you track your eyes again, the image on "3d Static Image" should remain static relative to you in 3d (it basically looks like you're looking out of a window)

### Position estimation and 3D scene rendering
* Run `positionserver.py` and `testfile.blend` (indirectly uses `3dposclient.py`)
* Estimates a person's position relative to the screen and uses it to render an explorable 3d environment based on their position.
* This experiment uses `UNIX Domain Sockets` for inter-process communication. It'll probably not work on non-UNIX-based environments.
* How to use:
    * Open `positionserver.py` and do the same procedures as in the last section (you can just copy the values over), then open `testfile.blend`, go to the text editor section and run the script
    * Now if you go to the 3d view screen and play the animation (forwards or backwards - what matters is the frames ticking) the camera position should be synchronised with your real position.
### Gaze direction estimation
* Run `gazedirection.py`
* Estimates the direction a person is looking at based on eye tracking and a third point (a high-contrast mark using a piece of paper or a marker pen dot on your forehead should be enough).
* How to use:
    * Open `gazedirection.py`, click on your eyes to track them then on the third reference tracker.
    * Look at the top left corner of the result screen and press "l", then at the bottom right and press "r".
    * There should now be a little red circle at approximately the position of your gaze if you keep your head at the same angle as when you calibrated the reference positions.



## Utilities

This repository has a couple utility functions to make it less redundant and more legible. Those are available at `utils.py`.

There, the following functions are implemented:

* `smooth(array:np.ndarray,newarray:np.ndarray,factor:float)->np.ndarray`
    * Used for smoothing out transitions between array states. It can be used, for example, for reducing tracking noise for a smoother experience, and it's internally used in the tracker for adapting its target image as time goes on.
    * Receives two n-dimensional arrays of the same shape and a smoothing parameter.
* `locmin(img:np.array,x:int,y:int,w:int,pimg:np.array,w2:int)->tuple`
    * Used for tracking high-contrast points at an image.
    * Receives an image, the position of the point you want to track, the width of the tracking area, the previously returned kernel (or None for the first iteration) and the width of the kernel to look for.
* `real_xyz_from_screen_xy(current_vector2d_screen:np.ndarray,screen_centre:np.ndarray,screen_length_at_reference_z:float,reference_z:float,real_length:float)->np.ndarray`
    * Calculates the real-world translation of the target relative to the camera (assuming the vector is parallel to the image plane).
    * Receives the current vector, the centre coordinates of the screen, the length the vector assumes at the real reference z, the reference z and the real length of the vector.
* `project_image_at_xyz(img:np.ndarray,xyz:np.ndarray,reference_z:float,target_window_shape:tuple,fixedscale:float,fixedtranslation:tuple,screen_pixel_density:float)`
    * Uses the real XYZ coordinates of the target to project an image that looks static to the viewer.
    * Receives the image you want to project, the real XYZ coordinates of the target, the reference real distance, the target window size, a fixed scaling factor (for scaling the "real" image), a fixed translation factor (for moving the "real" image in relation to the viewer) and the screen pixel density.