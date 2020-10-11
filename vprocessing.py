# --------------------------IMPORTS-------------------------- #


import cv2
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk, Image
import threading
import pyfftw
import os
from kernels import kernel


# ------------------------CONVOLUTION------------------------ #


def convolve(array_a, array_b):
    fft_array_a = pyfftw.builders.fft2(
        array_a, threads=os.cpu_count())
    fft_array_b = pyfftw.builders.fft2(
        array_b, s=array_a.shape, threads=os.cpu_count())
    ifft = pyfftw.builders.ifft2(
        fft_array_a()*fft_array_b(), threads=os.cpu_count())
    return np.real(ifft())

        
# -------------------------INTERFACE------------------------- #


class Application(tk.Frame):
    def __init__(self):
        self.root = tk.Tk()
        super().__init__(self.root)
        self.stream = None
        self.video_panel = None
        self.frame = None
        self.effect = kernel['identity']
        self.pack()
        self.create_widgets()
        self.init_screen()
        self.init_video_thread()

    def init_screen(self):
        self.root.wm_title("Video Processing")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def init_video_thread(self):
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.start()
        
    def create_widgets(self):
        self.effects_list = ttk.Combobox(self,
                                         values=list(map(lambda n: n, kernel)),
                                         state="readonly")
        self.effects_list.pack(side="top", fill=tk.BOTH, expand=1)
        self.effects_list.bind("<<ComboboxSelected>>", self.set_effect)
        self.effects_list.current(0)

    def set_effect(self, event):
        self.effect = kernel.get(list(kernel)[self.effects_list.current()])

    def video_loop(self):
        self.stream = cv2.VideoCapture(0)
        while not self.stopEvent.is_set():
            ret, self.frame = self.stream.read()
            if ret:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCR_CB)
                convolve_result = self.channel_apply_effect(0)
                image = cv2.merge((convolve_result, self.frame[:, :, 1], self.frame[:, :, 2]))
                image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2RGB)
                image = ImageTk.PhotoImage(Image.fromarray(image))
                self.update_image(image)
            else:
                break
        self.stream.release()

    def update_image(self, image):
        if self.video_panel is None:
            self.video_panel = tk.Label(image=image)
            self.video_panel.image = image
            self.video_panel.pack(
                side="bottom", fill=tk.BOTH, expand=1)
        else:
            self.video_panel.configure(image=image)
            self.video_panel.image = image

    def channel_apply_effect(self, i):
        return np.uint8(np.round(convolve(
            self.frame[:, :, i], self.effect)))

    def onClose(self):
        self.stopEvent.set()
        self.root.quit()


app = Application()
app.mainloop()
