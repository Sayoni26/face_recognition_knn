import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
cap = cv2.VideoCapture(0)
faceCasacade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

data=np.load("faces.npy")

X=data[:,1:].astype(int) #data points of the face
y=data[:,0] #names

model=KNeighborsClassifier(4)
model.fit(X,y)

face_list=[]
while True:
    ret, img = cap.read() #capturing the image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    faces = faceCasacade.detectMultiScale(gray)
    areas=[]
    for face in faces:
        x,y,w,h=face
        area=h*w
        areas.append((area,face))

    areas=sorted(areas,reverse=True)
    if len(areas)>0:
        face=areas[0][1]
        x,y,w,h=face
        face_img=gray[y:y+h,x:x+w]
        face_img=cv2.resize(face_img,(100,100))

        flat=face_img.flatten()
        res=model.predict([flat])
        
        na=np.array2string(res)
        print(na)
        l=len(na)
        print(type(na))
        
        for face in faces:            
            cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),5)
            cv2.putText(gray,na[2:(l-2)],(x,h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
            cv2.imshow("video",gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()

