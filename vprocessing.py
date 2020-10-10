# --------------------------IMPORTS-------------------------- #

import cv2
import numpy as np
import tkinter as tk

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

# ------------------------CONVOLUTION------------------------ #
