import cv2
def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3 , 5)
        
        if faces is ():
            return None
        lst =[]
        for (x,y,w,h) in faces:
            lst.append([y,y+h,x,x+w])
        return lst
    
    cap = cv2.VideoCapture(0)
    img_id = 0
    
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            imgs = face_cropped(frame)
            j=0
            for i in imgs:
                
                img_id+=1
                face = cv2.resize(frame[i[0]:i[1],i[2]:i[3]], (200,200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                #file_name_path = "data/"+"Ishwar."+str(img_id)+".jpg"
                # file_name_path = "dustbin2/"+"photo."+str(img_id)+'.jpg'
                # cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
            
                cv2.imshow(f"Cropped_Face{j}", face)
                if cv2.waitKey(100)==13 or int(img_id)==500:
                    break
                j+=1;
                
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed !!!")
generate_dataset()