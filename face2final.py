import cv2
import numpy as np
from os import path
cap = cv2.VideoCapture(0)
faceCasacade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# collection of data (name and pic)

# input name and take 50 snapshot 
name=input("input your name")
count=50

# list which will contain frames
face_list=[]

while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #converting rgb images to gray scale images
    faces = faceCasacade.detectMultiScale(gray) # detecting faces and reducing the width and height of your image
    # it returns a list of rectangles or boxes

    areas=[]

    for face in faces: # taking one frame(here box) at a time  

        x,y,w,h=face
        # x->x coordinate of the box
        # y-> y coordinate of the box
        # w-> width of the box 
        # h -> height of the box

        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),5) 
        # this function makes a rectangular box of black colour with thickness 5 
        # it starts from x,y and ends at x+w,y+h

        area=h*w #area of box

        areas.append((area,face)) # all the areas and the initial frame is appended in a list

    cv2.imshow("vedio",gray)  # shows the image

    areas=sorted(areas,reverse=True) #area is sorted from big to small

    if len(areas)>0: 
        face=areas[0][1] #take the frame
        x,y,w,h=face #unpack the x,y,w,h parameters
        face_img=gray[y:y+h,x:x+w] #region of interest
        face_img=cv2.resize(face_img,(100,100)) #resize the roi and it is 2-d

        face_list.append(face_img.flatten()) #flatten the 2-d image to make it 1-d
        count-=1 #keeping a count as we will take 50 frames
        print("loaded",50-count)
        if count<0:
            break

    if cv2.waitKey(1) & 0xFF == ord('q'): #stop capturing when presses the 'q' key
        break


face_list=np.array(face_list) #making numpy array of the 1-d image
name_list=np.full((len(face_list),1),name) #making a list of names
total=np.hstack((name_list,face_list)) #horizontally stacking the two np arrays

if path.exists("faces.npy"): #if faces.npy already exists
    data=np.load("faces.npy") #load the data
    data=np.vstack((data,total)) #and vertically stack the new data
else:
    data=total

np.save("faces.npy",data)

print(data.shape)

 
cap.release()
cv2.destroyAllWindows()

