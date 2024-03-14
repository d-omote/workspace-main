from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image, ImageTk

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

import numpy as np

import os

import tkinter as tk
import tkinter.filedialog

import styleTransfer
import poissonDiskSampling

##########################################################################

root = tk.Tk()
root.title('workspace')
root.geometry("1540x550")

#test
contentImageAdress = './content/other/dancing.jpg'
styleImageAdress  = './style/other/picasso.jpg'
transferedImage = Image.open(contentImageAdress)
pdsImage = Image.open(contentImageAdress)

# イベント
def openContentImage():
    global contentImageAdress
    global tk_content_img

    typ = [('JPGファイル', '*.jpg'), ('JPEGファイル', '*.jpeg'), ('PNGファイル', '*.png')] 
    dir = '.\\content\\now'
    contentImageAdress = tk.filedialog.askopenfilename(filetypes = typ, initialdir = dir)##ここ問題

    content_img = Image.open(contentImageAdress)
    content_img = content_img.resize((512, 512))
    tk_content_img = ImageTk.PhotoImage(content_img)
    img_width, img_height = content_img.size
    content_canvas = tk.Canvas(root, bg="#aaaaaa", height=img_height, width=img_width)
    content_canvas.place(x=0, y=0)
    content_canvas.create_image(0, 0, image=tk_content_img, anchor=tk.NW)
    
    print('コンテンツ画像を開きました')

def openStyleImage():
    global tk_style_img
    global styleImageAdress
    
    typ = [('JPGファイル', '*.jpg'), ('JPEGファイル', '*.jpeg'), ('PNGファイル', '*.png')] 
    dir = '.\\style'
    styleImageAdress = tk.filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    
    style_img = Image.open(styleImageAdress)
    style_img = style_img.resize((512, 512))
    tk_style_img = ImageTk.PhotoImage(style_img)
    img_width, img_height = style_img.size
    style_canvas = tk.Canvas(root, bg="#aaaaaa", height=img_height, width=img_width)
    style_canvas.place(x=512, y=0)

    style_canvas.create_image(0, 0, image=tk_style_img, anchor=tk.NW)
    print('スタイル画像を開きました')

def saveTransferedImage():
    fname = tk.filedialog.asksaveasfilename(title = "変換後の画像を保存",initialdir = ".\\result", filetype=[("PNG Image Files", ".png")],initialfile="transferedImage",defaultextension= "png")
    transferedImage.save(fname)
    print('スタイル変換画像を保存しました')

def savepdsImage():
    fname = tk.filedialog.asksaveasfilename(title = "PDS画像を保存",initialdir = ".\\result", filetype=[("PNG Image Files", ".png")],initialfile="pdsImage",defaultextension= "png")
    pdsImage.save(fname)
    print('PDS画像を保存しました')

def runStyleTransfer():
    global tk_transferedImage
    global transferedImage
    print('スタイル変換をします')
    transferedImage = styleTransfer.main(contentImageAdress, styleImageAdress)
    tk_transferedImage = ImageTk.PhotoImage(transferedImage)
    img_width, img_height = transferedImage.size
    transfered_canvas = tk.Canvas(root, bg="#aaaaaa", height=img_height, width=img_width)
    transfered_canvas.place(x=1024, y=0)
    transfered_canvas.create_image(0, 0, image=tk_transferedImage, anchor=tk.NW)
    print('スタイル変換しました')

def runPoissonDiskSampling():
    global tk_pdsImage
    global pdsImage
    
    pdsImage = poissonDiskSampling.main(transferedImage)
    tk_pdsImage = ImageTk.PhotoImage(pdsImage)
    img_width, img_height = pdsImage.size
    transfered_canvas = tk.Canvas(root, bg="#ffffff", height=img_height, width=img_width)
    transfered_canvas.place(x=1024, y=0)
    transfered_canvas.create_image(0, 0, image=tk_pdsImage, anchor=tk.NW)
    
    print('PDSしました')

def runStyleTransferAndPDS():
    runStyleTransfer()
    runPoissonDiskSampling()

#メニュー
menu = tk.Menu(root)
root.config(menu=menu)

subMenu = tk.Menu(menu)
menu.add_cascade(label='ファイル', menu=subMenu)

subMenu.add_command(label='コンテンツ画像を開く', command=openContentImage)
subMenu.add_command(label='スタイル画像を開く', command=openStyleImage)
subMenu.add_command(label='変換後の画像を保存', command=saveTransferedImage)
subMenu.add_command(label='pds後の画像を保存', command=savepdsImage)

editMenu = tk.Menu(menu)
menu.add_cascade(label='編集', menu=editMenu)

editMenu.add_command(label='スタイル変換', command=runStyleTransfer)
editMenu.add_command(label='poisson disk sampling', command=runPoissonDiskSampling)
editMenu.add_command(label='スタイル変換+pds', command=runStyleTransferAndPDS)
#editMenu.add_command(label='絵画に合成', command=runStyleTransfer)

root.mainloop()
