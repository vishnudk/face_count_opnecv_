import numpy as np
import cv2
# from digit_reco_cnn import *
# from speech2 import *
prev=(0,0)
flag=0
result=(0,0)
def main():
    cap = cv2.VideoCapture(0)
    
    i=0
    global flag
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.flip(frame, 1)  
        # gray=frame
        window_name = 'Image'
        i=i+1
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        # org 
        org = (30, 440) 
        
        # fontScale 
        fontScale = 1
       
        # Blue color in BGR 
        color = (255, 4, 0) 
        
        # Line thickness of 2 px 
        thickness = 1
        
        # gray=draw_circle(gray)
        # gray=draw_line(gray)
        # Display the resulting frame
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # face_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') 
        # haar_cascade_face = cv2.CascadeClassifier('/data/haarcascade_frontalface_alt.xml')
        gray, flag = detect_faces(flag,face_cascade, gray)
        print(flag)
        # if flag==1:
        #     flag=flag+1
        #     # speak_the_text(" hello !")
        #     print("face")
        text='number of people :'+str(flag)
        text=str(text)
        # Using cv2.putText() method 
        gray = cv2.putText(gray,text, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
def draw_circle(image):
    # center_coordinates =(find_color(image),40)
    arr=np.asarray(image)
    center_coordinates =(40,40)
    # tmp=find_color(image)
    # Radius of circle 
    radius = 20
    # speak_the_text("hello vishnu")
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 2
    
    # Using cv2.circle() method 
    # Draw a circle with blue line borders of thickness of 2 px 
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image 
def find_color(img):
    prev=(0,0)
    result=(0,0)
    arr=np.asarray(img)
    # print(arr.shape)
    upperLimit=np.asarray([235,235,235])
    lowerLimit=np.asarray([229,229,229])
    # for y in range(arr.shape[0]):
    #     forx in range(arr.shape[1]):
    #         if (arr[y][x]<=upperLimit).all() and (arr[y][x]>=lowerLimit).all():
    #            print(x)
    #            print("==========")
    #            print(y)
    #            return (x,y) 
    td_array=arr.flatten()
    td_array=np.resize(td_array,(307200,3))
    for clr in range(td_array.shape[0]-1):
        if ( td_array[clr]<=upperLimit).all() and ( td_array[clr]>=lowerLimit).all():
           prev=result
           result=np.where((arr== td_array[clr]).all())
           print(result)
           return result
    print(td_array.shape)
    # print(td_array.shape)
    return prev
def edge(frame):
     # Convert to HSV for simpler calculations 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
      
    # Calcution of Sobelx 
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5) 
      
    # Calculation of Sobely 
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5) 
      
    # Calculation of Laplacian 
    laplacian = cv2.Laplacian(frame,cv2.CV_64F) 
      
    cv2.imshow('sobelx',sobelx) 
    cv2.imshow('sobely',sobely) 
    cv2.imshow('laplacian',laplacian) 
def draw_line(image):
    # start_point2=(240,40)
    # End coordinate, here (250, 250) 
    # represents the bottom right corner of image 
    dim=image.shape
    x_dim=int(dim[1]/2)
    start_point = (x_dim, 0) 
    end_point = (x_dim,int(dim[0])) 
    # Green color in BGR 
    color = (0, 255, 0) 
    # color1=(0,255,255)
    # Line thickness of 9 px 
    thickness = 2
    # end_point2=(300,300)
    # Using cv2.line() method 
    # Draw a diagonal green line with thickness of 9 px 
    image1 = cv2.line(image, start_point, end_point, color, thickness) 
    # image = cv2.line(image1, start_point2, end_point2, color1, thickness) 
    return image
def detect_faces(flag,cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    if faces_rect!=():
        flag=flag+1
        # print(faces_rect.shape[0])
        no_of_people=faces_rect.shape[0]
    else:
        flag=0
        no_of_people=0
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)
    
    return image_copy,no_of_people
main()