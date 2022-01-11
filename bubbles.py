import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from collections import Counter
import os

def round_sig(x, sig=3):
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


'''
Set parameters
'''
min_rad = 0.6  # minimum radius of bubble (relative)
max_rad = 250  # maximum radius of bubble (relative)
n_frames = 350  # number of frames supplied
video_fps = 30  # original video FPS
sampling_fps = 10  # sampling rate of frames
view_size = 20  # size of view (in mm)
offset = 60  # offset to eliminate final drop

directory="C:/Users/tilen/Desktop/602/"

plot=False
savegif = False  # set to False to skip saving GIF
run = False #set to False to skip image processing (use only if you change plotting parameters)

if run == True:
    size = []
    size_max = []
    size_min = []
    num = []
    images = []
    counter=0
    size_all=[]

    for filename in os.listdir(directory):
        '''
        Change to directory of frames
        '''
        img = cv2.imread(
            f'{directory}{filename}', cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        

        contours, hierarchy = cv2.findContours(
            image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            boundRect[i] = cv2.boundingRect(contours[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours[i])
        com = Counter(hierarchy[0][:, 3]).most_common(1)[0][0]
        if com == -1 and len(Counter(hierarchy[0][:, 3]))>1:
            com = Counter(hierarchy[0][:, 3]).most_common(2)[1][0]
        for i in range(len(contours)):
            if radius[i] < 500 and hierarchy[0][i][3] == com:
                cv2.circle(img, (int(centers[i][0]), int(
                    centers[i][1])), int(radius[i]), (0, 255, 0), 2)

        images.append(img)
        if len(contours) > 0:
            x = np.array(radius)
            size.append(
                (np.mean(x[np.logical_and(x < max_rad, x > min_rad, hierarchy[0][:, 3] == com)]))/width*view_size)
            size_max.append(
                (np.max(x[np.logical_and(x < max_rad, x > min_rad, hierarchy[0][:, 3] == com)]))/width*view_size)
            size_min.append(
                (np.min(x[np.logical_and(x < max_rad, x > min_rad, hierarchy[0][:, 3] == com)]))/width*view_size)
            num.append(
                len(x[np.logical_and(x < max_rad, x > min_rad, hierarchy[0][:, 3] == com)]))
            size_all.append((x[np.logical_and(x < max_rad, x > min_rad, hierarchy[0][:, 3] == com)])/width*view_size)
        else:
            size.append(0)
            size_max.append(0)
            size_min.append(0)
            num.append(0)
        counter+=1
    if savegif == True:
        print('Saving GIF...')
        '''
        Change to directory you want the GIF saved
        '''
        imageio.mimsave('C:/Users/tilen/Pictures/movie.gif',
                        images)
        
    ys = movingaverage(size, 10)
    ysu = movingaverage(size_max, 10)
    ysl = movingaverage(size_min, 10)
    size=size[:300]
    num=num[:300]
    ys=ys[:300]
    ysu=ysu[:300]
    ysl=ysl[:300]
    xs = (np.linspace(0, len(size), 300)/(video_fps/sampling_fps))

if plot==True:
    print('Plotting graphs...')
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax3.set_xlabel('Time [s]')
    ax1.set_ylabel('Max. Radius [mm]')
    ax2.set_ylabel('Number')
    ax3.set_ylabel('Bubble formation')
    
    ax1.plot(xs, size, 'k.')
    ax1.plot(xs, ysu)
    ax1.plot(xs, ysl, '--', color='cyan')
    ax1.plot(xs, ys)
    ax2.bar(xs, num)
    ax1.text(0.65, 0.1, f"Avg. Size: {round_sig(np.mean(size))}mm",transform=ax1.transAxes, bbox=dict(facecolor='none', edgecolor="black", alpha=0.5))
    plt.show()

print("Avg Size: ",np.mean(size))
print("Max Size: ",np.max(ysu))
print("Max Number: ",np.max(num))
print("Avg Number: ",np.mean(num))