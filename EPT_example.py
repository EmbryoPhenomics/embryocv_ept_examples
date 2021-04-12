# Necessary imports --------------------------------------------------------------
# Retrieval of source data
import wget

# Image handling
from skimage.util import view_as_blocks
import numpy as np 
import vuba
import cv2
import math

# Power spectral analysis
from scipy import signal

# Plotting 
import matplotlib.pyplot as plt

# Parameters ----------------------------------------------------------------------
# Grid size for EPT calculation 
blocksize = 8

# Path to footage to analyse
path = 'https://zenodo.org/record/4645805/files/20C_A1_10d.avi?download=1'

# Computation of EPTs -------------------------------------------------------------
def crop(frame):
    '''Crop an image according to the blocksize required for EPT calculation.'''
    # Floor used so new image size is always below the original resolution
    new_x, new_y = map(lambda v: math.floor(v/blocksize)*blocksize, frame.shape) 
    return frame[:new_y, :new_x]

# Retrieve footage and read in frames
path = wget.download(path)
video = vuba.Video(f'./{path}')
frames = video.read(0, len(video), grayscale=True)

# Compute mean pixel values 
x,y = video.resolution
block_shape = tuple(map(int, (x/blocksize, y/blocksize)))
mpx = np.asarray([view_as_blocks(frame, block_shape).mean(axis=(2,3)) for frame in map(crop, frames)])

# Compute power spectral data
epts = np.empty((blocksize, blocksize, 2, int(len(video)/2)+1))
for i in range(blocksize):
    for j in range(blocksize):
        epts[i,j,0,:], epts[i,j,1,:] = signal.welch(mpx[:,i,j], fs=video.fps, scaling='spectrum', nfft=len(mpx))

# Plot results --------------------------------------------------------------------
# Mean pixel values
fig, axs = plt.subplots(blocksize, blocksize, sharex=True)
fig.subplots_adjust(left=0.08, right=0.98, wspace=0.1)

for i in range(blocksize):
    for j in range(blocksize):
        ax = axs[i,j]
        ax.set_autoscale_on(True)
        ax.get_yaxis().set_visible(False)
        ax.plot(mpx[:,i,j], linewidth=0.5)

fig.suptitle('Mean pixel values')
fig.text(0.5, 0.04, 'Frame', ha='center')
fig.text(0.04, 0.5, 'Mean pixel value (scaled independantly per block)', va='center', rotation='vertical')
plt.show()

# Energy proxy traits
fig, axs = plt.subplots(blocksize, blocksize, sharex=True)
fig.subplots_adjust(left=0.08, right=0.98, wspace=0.1)

for i in range(blocksize):
    for j in range(blocksize):
        ax = axs[i,j]
        ax.set_autoscale_on(True)
        ax.get_yaxis().set_visible(False)
        ax.plot(epts[i,j,0,:], np.log(epts[i,j,1,:]), linewidth=0.5)

fig.suptitle('Energy proxy trait spectra')
fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center')
fig.text(0.04, 0.5, 'Energy', va='center', rotation='vertical')
plt.show()

# Close the video handler
video.close()
