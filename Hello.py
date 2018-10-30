# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import * 
import os as os
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import progressbar
import argparse

def create_image(i,j):
    image = Image.new("RGB",(i,j),"white")
    return image
def ImproveData(PathToFile,Pb,Pb2):
    cap = cv2.VideoCapture(PathToFile)
    videoName = os.path.basename(root.filename)
    videoName = videoName[:-4]
    img = []
    try:
        if not os.path.exists(videoName):
            os.makedirs(videoName+'/OriginalData')
            os.makedirs(videoName+'/DenoiseData')
            os.makedirs(videoName+'/ImageBordersData')
            os.makedirs(videoName+'/FinalData')
    except OSError:
        print ('Error: Creating directory of data')
    currentFrame = 0
    T=bool(1)
    while(T):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Saves image of the current frame in png file
        name = './'+videoName+'/OriginalData/frame' + str(currentFrame) + '.png'
        # name = './OriginalData/frame' + str(currentFrame) + '.png'
        cv2.imwrite(name, frame)
        if os.path.getsize(name) == 0:
            T=bool(0)  # To stop duplicate images
            os.remove(name)  
        else:
            currentFrame += 1
            img.append(frame)
    CountOfImages=currentFrame-1
    # Denoising  
    st = (1/(CountOfImages-4)*100)
    for i in range(2,CountOfImages-1):
        dst = cv2.fastNlMeansDenoisingMulti(img,i,5,None,4, 7,35)
        #name = './DenoiseData/DenoisedFrame' + str(i) + '.png'
        name = './'+videoName+'/DenoiseData/DenoisedFrame' + str(i) + '.png'
        cv2.imwrite(name, dst)
        pb.step(st) 
        pb.update() 
    # Detection of borders in image
    Dic_of_Pixel={}
    Original_Image = Image.open('./'+videoName+'/DenoiseData/DenoisedFrame5.png')
    global width, height
    width, height = Original_Image.size
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    for k in range(2,CountOfImages-1):
        name = './'+videoName+'/DenoiseData/DenoisedFrame' + str(k) + '.png'
        nameBorder = './'+videoName+'/ImageBordersData/BorderFrame' + str(k) + '.png'
        Grad(name,nameBorder,Dic_of_Pixel)
        nameFinal = './'+videoName+'/FinalData/FinalFrame' + str(k) + '.png'
        # Sharpen of image
        Sharpened(name,nameFinal,Dic_of_Pixel,kernel)
        Dic_of_Pixel.clear()
        pb2.step(st) 
        pb2.update()
    
    # Creat video (in .mov and .mp4 format) from images in FinalData, DenoiseData, ImageBordersData
    image_folder = './'+videoName+'/FinalData/'
    video_name = './'+videoName+'/DenoisedVideoFinalData.mov'
    video_name2 = './'+videoName+'/DenoisedVideoFinalData.mp4'
    image_folderDen = './'+videoName+'/DenoiseData/'
    video_nameDen = './'+videoName+'/DenoisedVideoDenoisedData.mov'
    video_nameDen2 = './'+videoName+'/DenoisedVideoDenoisedData.mp4'
    image_folderBor = './'+videoName+'/ImageBordersData/'
    video_nameBor = './'+videoName+'/BordersVideo.mov' 
    video_nameBor2 = './'+videoName+'/BordersVideo.mp4'
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    imagesDen = [img for img in os.listdir(image_folderDen) if img.endswith(".png")]
    imagesBor = [img for img in os.listdir(image_folderBor) if img.endswith(".png")]
    
    imag=[]
    imagDen=[]
    imagBor=[]

    for i in range(2,CountOfImages-1):
        imag.append('FinalFrame' + str(i) + '.png')
        imagDen.append('DenoisedFrame' + str(i) + '.png')
        imagBor.append('BorderFrame' + str(i) + '.png')
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, cap.get(cv2.CAP_PROP_FPS), (width,height))
    video2 = cv2.VideoWriter(video_name2, -1, cap.get(cv2.CAP_PROP_FPS), (width,height))
    video3 = cv2.VideoWriter(video_nameDen, -1, cap.get(cv2.CAP_PROP_FPS), (width,height))
    video4 = cv2.VideoWriter(video_nameDen2, -1, cap.get(cv2.CAP_PROP_FPS), (width,height))
    video5 = cv2.VideoWriter(video_nameBor, -1, cap.get(cv2.CAP_PROP_FPS), (width,height))
    video6 = cv2.VideoWriter(video_nameBor2, -1, cap.get(cv2.CAP_PROP_FPS), (width,height))
    for image in imag:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        video2.write(cv2.imread(os.path.join(image_folder, image)))
    for image in imagDen:    
        video3.write(cv2.imread(os.path.join(image_folderDen, image)))
        video4.write(cv2.imread(os.path.join(image_folderDen, image)))
    for image in imagBor:    
        video5.write(cv2.imread(os.path.join(image_folderBor, image)))
        video6.write(cv2.imread(os.path.join(image_folderBor, image)))
    cv2.destroyAllWindows()
    video.release()
    video2.release()
    video3.release()
    video4.release()
    video5.release()
    video6.release()
def Brightness(RGB):
    R,G,B = RGB
    J = int(sum([R,G,B])/3)
    return J
def Derivation(image,i,j):
    # Derivace ve vodorovném směru
    if i == 0:
        Jas1 = Brightness(image.getpixel((i+1,j)))
        Jas2 = Brightness(image.getpixel((i,j)))
        derX = Jas1 - Jas2
    elif i == (width-1):
        Jas1 = Brightness(image.getpixel((i,j)))
        Jas2 = Brightness(image.getpixel((i-1,j)))
        derX = Jas1 - Jas2
    else:
        Jas1 = Brightness(image.getpixel((i+1,j)))
        Jas2 = Brightness(image.getpixel((i-1,j)))
        derX = (Jas1-Jas2)/2
    # Derivace ve svislém směru
    if j == 0:
        Jas1 = Brightness(image.getpixel((i,j+1)))
        Jas2 = Brightness(image.getpixel((i,j)))
        derY = Jas1 - Jas2
    elif j == (height-1):
        Jas1 = Brightness(image.getpixel((i,j)))
        Jas2 = Brightness(image.getpixel((i,j-1)))
        derY = Jas1 - Jas2
    else:
        Jas1 = Brightness(image.getpixel((i,j+1)))
        Jas2 = Brightness(image.getpixel((i,j-1)))
        derY = (Jas1 - Jas2)/2
    return [derX,derY]
def Grad(path,path2,MyDict):
    max=0
    Original_Image = Image.open(path)
    #width, height = Original_Image.size 
    Border_Image = create_image(width,height)
    for i in range(0,width):
        for j in range(0,height):
            derX, derY = Derivation(Original_Image,i,j)
            norm = math.sqrt(derX*derX+derY*derY)
            if norm > 500: 
                MyDict[j*width+i] = [i,j]
            if norm > max:
                max = norm
    pixels = Border_Image.load()
    for i in range(0,width):
        for j in range(0,height):
            derX, derY = Derivation(Original_Image,i,j)
            norm = math.sqrt(derX*derX+derY*derY)
            Jas = int((norm/max)*255)            
            pixels[i,j]=(Jas,Jas,Jas)
    Border_Image.save(path2)
def Sharpened(path,nameF,Dic,Core):
    FinalImage = create_image(width,height)
    pixels = FinalImage.load()
    DenImage = Image.open(path)
    for i in range(0,width):
        for j in range(0,height):
            if j*width+i in Dic:
                B = 0
                for i2 in range(-1,2):
                    for j2 in range(-1,2):
                        B = B + Brightness(DenImage.getpixel((i+i2,j+j2)))*Core(1+i2,1+j2)   
                pixels[i,j]=(B,B,B)
            else:
                B = Brightness(DenImage.getpixel((i,j)))
                pixels[i,j]=(B,B,B)
    FinalImage.save(nameF) 

root = tk.Tk()
root.filename = filedialog.askopenfilename(initialdir ="C:", title = "Select file", filetypes = (("all files","*.*"),("avi files","*.avi")))
os.path.basename(root.filename)
w = tk.Label(root, text="I am working with your data.")
w.pack()
pb = ttk.Progressbar(root, orient="horizontal", length=100, mode="determinate")
pb.pack()
pb2 = ttk.Progressbar(root, orient="horizontal", length=100, mode="determinate")
pb2.pack()
width = 0
height = 0 
B=tk.Button(root, text="Read data", command=ImproveData(root.filename,pb,pb2))
B.pack()
w1 = tk.Label(root, text="Your video has been denoised.",bg="green",fg="black")
w1.pack()
root.mainloop()
# When everything done, release the capture
cv2.destroyAllWindows()