#!/usr/bin/python
from Tkinter import *
from tkFileDialog import askopenfilename
import Image, ImageTk
from imh import *
import cv2

if __name__ == "__main__":
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    File = askopenfilename(parent=root, initialdir="/Users/Jon/Documents/College/Research/HealyLab/healy_cellsorter0",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.open(File))
    
    print File
    cv_img = cv2.imread(File)
    hsv_cv_img = cv2.cvtColor(cv_img.copy(), cv2.COLOR_BGR2HSV)


    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        print (event.x,event.y)
        print "HSV VALUES: " + str(hsv_cv_img[event.y, event.x])
    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()