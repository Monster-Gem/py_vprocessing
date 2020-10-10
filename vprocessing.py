# --------------------------IMPORTS-------------------------- #


import cv2
import numpy as np
import tkinter as tk
import pyfftw
import os
from kernels import kernel


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
# app.mainloop()


# ------------------------CONVOLUTION------------------------ #

def convolve(array_a, array_b):
    fft_array_a = pyfftw.builders.fft2(array_a, threads=os.cpu_count())
    fft_array_b = pyfftw.builders.fft2(array_b, s=array_a.shape,
                                       threads=os.cpu_count())
    ifft = pyfftw.builders.ifft2(fft_array_a()*fft_array_b(),
                                 threads=os.cpu_count())
    return np.real(ifft())[1:, 1:]


# -------------------------VIDEO I/O------------------------- #


cam = cv2.VideoCapture(0)
if (not cam.isOpened()):
    print("Error opening the camera")
while(cam.isOpened()):
    ret, frame = cam.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        conv = np.uint8(np.round(convolve(gray, kernel['sharpen'])))
        cv2.imshow('Frame', conv)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cam.release()
cv2.destroyAllWindows()
