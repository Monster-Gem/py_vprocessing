# --------------------------IMPORTS-------------------------- #

import cv2
import numpy as np
import tkinter as tk
import pyfftw
import os

# -------------------------INTERFACE------------------------- #


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        
        self.effects_list = tk.Listbox(self)
        self.effects_list.pack(side="left", fill=tk.BOTH, expand=1)
        
        self.video = tk.Canvas(self)
        self.video.pack(side="right", fill=tk.BOTH, expand=1)


root = tk.Tk()
root.title("Video Processing")
app = Application(master=root)
app.mainloop()

# -------------------------VIDEO I/O------------------------- #

cam = cv2.VideoCapture(0)
if (cam.isOpened()== False): 
    print("Error opening the camera")
while(cam.isOpened()):
    ret, frame = cam.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        conv = np.uint8(np.round(convolve(gray, kernel['sharpen'])))
        cv2.imshow('Frame', conv)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break
cam.release()
cv2.destroyAllWindows()

# ------------------------CONVOLUTION------------------------ #

kernel = {
    'identity': np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float),
    'edge detection': np.array([[1,0,-1],[0,0,0],[-1,0,1]], dtype=float),
    'laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float),
    'laplacian w/ diagonals': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float),
    'laplacian of gaussian': np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]], dtype=float),
    'scharr': np.array([[-3, 0, 3],[-10,0,10],[-3, 0, 3]], dtype=float),
    'sobel edge horizontal': np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float),
    'sobel edge vertical': np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float),
    'line detection horizontal': np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float),
    'line detection vertical': np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=float),
    'line detection 45°': np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype=float),
    'line detection 135°': np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype=float),
    'box blur': (1/9)*np.ones((3,3), dtype=float),
    'gaussian blur 3x3': (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float),
    'gaussian blur 5x5': (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
    'sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
    'unsharp masking': (-1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
}

def convolve(array_a, array_b):
    fft_array_a = pyfftw.builders.fft2(array_a, threads=os.cpu_count())
    fft_array_b = pyfftw.builders.fft2(array_b, s=array_a.shape, threads=os.cpu_count())
    ifft = pyfftw.builders.ifft2(fft_array_a()*fft_array_b(), threads=os.cpu_count())
    return np.real(ifft())[1:,1:]
