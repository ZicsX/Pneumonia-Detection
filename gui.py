import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
from keras.models import model_from_json




json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("final_model.h5")
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



top = Tk()
top.title("CXR")

canvas  = Canvas(top, width = 500, height = 500)
canvas.pack()

var = StringVar()
widget = Label(top,  textvariable=var, font=('arial',16,'bold'),bd=10,bg="purple1",justify='right').pack()
var.set("RESULT_LABLE")

myimg0 = Image.open("li.jpg")
myimg0 = myimg0.resize((500, 500), Image.ANTIALIAS)
myimg = ImageTk.PhotoImage(myimg0)
canvas.create_image(0,0, anchor="nw", image=myimg)


def openfile():

	global myimg
	top.filename =  filedialog.askopenfilename(initialdir="C:/Users/zics/Desktop/py/CXR-PN/test/",title = "Select file",filetypes = (("jpeg files","*.jpeg"),("all files","*.*")))
	f = top.filename
	
	
	img = plt.imread(f)
	img = cv2.resize(img, (150,150))
	img = np.dstack([img, img, img])
	img = img.astype('float32') / 255
	img = [img,img]
	img = np.array(img)

	res = loaded_model.predict(img)
	result = int(np.round(res[0]))

	if (result == 0):
		var.set("NORMAL")
	else:
		var.set("PNEUMONIA")
	
	
	myimg0 = Image.open(f)
	myimg0 = myimg0.resize((500, 500), Image.ANTIALIAS)
	myimg = ImageTk.PhotoImage(myimg0)
	
	canvas.create_image(0,0, anchor="nw", image=myimg)

	
B = Button(top,padx=16,pady=8,bd=16,fg="black",font=('arial',16,'bold'),width=10,text="Load Image",bg="purple1",command=openfile)

#label = Label( top, textvariable=var, font=('arial',16,'bold'),bd=10,insertwidth=4,bg="purple1",justify='right')

#label.pack()
B.pack()
top.mainloop()


