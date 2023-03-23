import os
import tkinter as tk 
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf 

#load the model
model = tf.keras.models.load_model('trained model.h5',compile=False)
model.compile()
 
#x = [[3.]]
#y = [[4.]]
#print('Result: {}'.format(tf.matmul(x, y)))

#define the window 
window = tk.Tk()
window.geometry("450x450")
window.title("Dogs and Cats recognition")
#define the grid 
frame = tk.Frame(window)
frame.columnconfigure(0,weight=1)
frame.columnconfigure(1,weight=1)
frame.columnconfigure(2,weight=1)
frame.columnconfigure(3,weight=1)
#define title
title = tk.Label(window,text="please select a picture to see if its a dog or a cat",font=('Arial',14))
title.pack()
#define image label
filepath = "default image.jpg"
filepatWrapper = [filepath] 
image = Image.open(filepath)
n_image = image.resize((300,300))
photo = ImageTk.PhotoImage(n_image)
imgLabel = tk.Label(frame,image=photo)
imgLabel.photo = photo  
imgLabel.grid(row=2,column=1)
#define user label
massege = "hello world! "
massegeWrapper = [massege]
userLabel = tk.Label(frame,text=massege) 
userLabel.grid(row=3,column=2)
               
#define select image button 
def selectImage(filepatWrapper) :
     f_types = [('Jpg Files', '*.jpg'), ('PNG Files','*.png')]
     file = filedialog.askopenfile(filetypes=f_types)
     filepatWrapper[0] = os.path.abspath(file.name)
     if file:
      filepath = filepatWrapper[0]
      image = Image.open(filepath)
      n_image = image.resize((300,300))
      photo = ImageTk.PhotoImage(n_image)
      imgLabel.config(image=photo)
      imgLabel.photo = photo  
imageButton = tk.Button( frame,text='Select an Image', command=lambda:selectImage(filepatWrapper))
imageButton.grid(row=4,column=0)
#define evaluate button  
def evaluateImage():
    vclass = " " 
    img = tf.keras.utils.load_img(filepatWrapper[0], target_size=(180,180))
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    type = predictions.argmax(axis=-1)
    if(type==0):
        vclass = "dog"
    else:
        vclass = "cat"    

    userLabel.config(text="This image most likely belongs to {}  with a {:.2f} percent confidence."
    .format(vclass,100 * np.max(score)))

   
evaluateButton = tk.Button(frame,text="evaluate image",command=lambda:evaluateImage())
evaluateButton.grid(row=4,column=1)
  
   




frame.pack()
window.mainloop() 


