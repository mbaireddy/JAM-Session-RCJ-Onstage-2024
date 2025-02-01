######################
# RCJ OnStage 2024 - processing the video from the cam and piano finger control
######################
#onstage
#middlest octave values==============================================================================================

#CORRECT UPDATED VALS FOR TOWER PROS (7/3)
C_START = 170
C_PRESS = 90


D_START = 165
D_PRESS = 95


E_START = 40
E_PRESS = 120


F_START = 170
F_PRESS = 90


G_START = 158
G_PRESS = 90

A_START = 40
A_PRESS = 110

B_START = 165
B_PRESS = 90



'''
C_START = 20
C_PRESS = 83

D_START = 40
D_PRESS = 123

E_START = 180
E_PRESS = 90

F_START = 70
F_PRESS = 122

G_START = 0
G_PRESS = 60

A_START = 140
A_PRESS = 60

B_START = 60
B_PRESS = 140
'''

import pickle
import mediapipe as mp

import cv2
import numpy as np
import time
import board
import threading
import queue

import serial

from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from picamera2 import Picamera2

q = queue.Queue()


i2c = board.I2C()  # uses board.SCL and board.SDA
# i2c = busio.I2C(board.GP1, board.GP0)    # Pi Pico RP2040

# Create a simple PCA9685 class instance.
pca = PCA9685(i2c)
# You can optionally provide a finer tuned reference clock speed to improve the accuracy of the
# timing pulses. This calibration will be specific to each board and its environment. See the
# calibration.py example in the PCA9685 driver.
# pca = PCA9685(i2c, reference_clock_speed=25630710)
pca.frequency = 50
picam2 = Picamera2(0)

config = picam2.create_preview_configuration(lores={"size": (640, 480)})
picam2.configure(config)
picam2.start()

model_dict = pickle.load(open('./model.p', 'rb')) 
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#creates a dictionary... character pressed: (piano key, servo num)
pause = 'P'
labels_dict = {
    0: ('C', servo.Servo(pca.channels[1], min_pulse=544, max_pulse=2400)  ),
    1: ('D', servo.Servo(pca.channels[2], min_pulse=544, max_pulse=2400)  ),
    2: ('E', servo.Servo(pca.channels[3], min_pulse=544, max_pulse=2400)  ),
    3: ('F', servo.Servo(pca.channels[4], min_pulse=544, max_pulse=2400)  ),
    4: ('G', servo.Servo(pca.channels[5], min_pulse=544, max_pulse=2400)  ),
    5: ('A', servo.Servo(pca.channels[6], min_pulse=544, max_pulse=2400)  ),
    6: ('B', servo.Servo(pca.channels[7], min_pulse=544, max_pulse=2400)  ),
    7: (pause, servo.Servo(pca.channels[8], min_pulse=544, max_pulse=2400)),
    8: ('L', servo.Servo(pca.channels[8], min_pulse=544, max_pulse=2400))}

last_servo = 0 #tracks milliseconds since the last servo was pressed

def start_servo():
    sleepytime=0.4
    
    labels_dict[0][1].angle = C_START
    time.sleep(sleepytime)
    
    labels_dict[1][1].angle = D_START
    time.sleep(sleepytime)
    
    labels_dict[2][1].angle = E_START
    time.sleep(sleepytime)
    
    labels_dict[3][1].angle = F_START
    time.sleep(sleepytime)
    
    labels_dict[4][1].angle = G_START
    time.sleep(sleepytime)

    labels_dict[5][1].angle = A_START
    time.sleep(sleepytime)
    
    labels_dict[6][1].angle = B_START
    time.sleep(sleepytime)
    
    
def servo_play_note(item):
    sleepytime = 0.4
    print(f'servo note({item[0]})')
    print(f'sleeping {sleepytime} second')
    #servo movement starts here                
    servooo = item[1]
    letter = item[0]
    print(letter, servooo)
    
    if(letter == 'C'):
        servooo.angle = C_PRESS  #right servo if looking from the back
        time.sleep(sleepytime)
        servooo.angle = C_START
        time.sleep(sleepytime)
    
    elif(letter == 'D'):    
        #D===================================
        servooo.angle = D_PRESS  #recked
        time.sleep(sleepytime)
        servooo.angle = D_START
        time.sleep(sleepytime)
        
        
    elif(letter == 'E'):        
        #E===================================
        servooo.angle = E_PRESS #rec
        time.sleep(sleepytime)
        servooo.angle = E_START
        time.sleep(sleepytime)
 
    elif(letter == 'F'): 
        #F====================================
        servooo.angle = F_PRESS
        time.sleep(sleepytime)
        servooo.angle = F_START
        time.sleep(sleepytime)

    
    elif(letter == 'G'): 
        #G====================================
        servooo.angle = G_PRESS
        time.sleep(sleepytime)
        servooo.angle = G_START
        time.sleep(sleepytime)

    elif(letter == 'A'):
        #A===========================================
        servooo.angle = A_PRESS
        time.sleep(sleepytime)
        servooo.angle = A_START
        time.sleep(sleepytime)

    elif(letter == 'B'):
        #A===========================================
        servooo.angle = B_PRESS
        time.sleep(sleepytime)
        servooo.angle = B_START
        time.sleep(sleepytime)
        
    elif(letter == 'L'):
        sleepytime=0.4
        print("hello")
        
        labels_dict[5][1].angle = A_START
        time.sleep(1)
        labels_dict[1][1].angle = D_START
        time.sleep(sleepytime)
        
        labels_dict[5][1].angle = A_PRESS
        labels_dict[1][1].angle = D_PRESS
        time.sleep(sleepytime)
      
        #makes sures the servos don't run too close to each other repeatedly
    #time.time() * 1000 gets the current miliseconds its been since last called
#     if((round(time.time() * 1000)) - last_servo > 1000): #difference between the milliseconds the servos last ran and right now
#         print("servo ran")

#     for i in range(180): 
#         servooo.angle = 180 - i
#         time.sleep(sleepytime)
#          last_servo = round(time.time() * 1000)
#     
#      else:
#          continue
            
    
old_note = 'z'
def worker_thread():
    print("in worker thread")
    global old_note
    while True: 
        item = q.get()
        note = item[0]
        servonum = item[1]
        if old_note != note:
            old_note = note
            
            '''
            current_secs = datetime.now()
            current_secs = current_secs.second + ((current_secs.microsecond)/1000000)
            
            timeToWait = int(current_secs + 0.99999999) - current_secs
            print(timeToWait)
            time.sleep(timeToWait)
            '''

            servo_play_note(item)
        q.task_done()

threading.Thread(target=worker_thread, daemon=True).start()

counter = [0,0,0,0,0,0,0,0]

#def predictChar():     #  "predictChar shouldn't be a function that's called, it should be constantly running to constantly be grabbing frames" - Dennis

start_servo()
print("servos started")

ser = serial.Serial('/dev/ttyAMA0', 115200)

while (True):
    
    data_aux = []
    x_ = []
    y_ = []
    yuv420 = picam2.capture_array("lores") 
    frame = cv2.cvtColor(yuv420, cv2.COLOR_YUV420p2RGB)
    #cv2.imshow('frame', frame)
    
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks: #getting live landmarks from camera
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks( #drawing landmarks on top of the camera
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks: #iterate through the landmarks and create the array
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)]) #use the imported model to predict the character, convert to nparray to use with prediction
            print(prediction)
            
            pred_val = int(prediction[0])
            print(pred_val)
                #print(labels_dict.keys())
                #print(int(prediction[0]))
                #predicted_tuple = labels_dict[int(prediction[0])] #prediction is a list of 1 element, and use this to access dict "labels" & get the actual character
    #             cv2.putText(img,'OpenCV',(10,500), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
    #             cv2.imshow('frame', frame)
                #return(predicted_tuple)
                

                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) #draw a rectangle around the hand
            counter[7] = 0
            max_count = max(counter)
            max_index = counter.index(max_count)
                
            predicted_obj  = labels_dict[max_index]
            
            if predicted_obj is not None:
                counter[pred_val] = counter[pred_val] + 1
                


                if predicted_obj[0] != pause and max_count >= 2: 
                    print("putting")
                    q.put(predicted_obj)
                    predicted_character = predicted_obj[0]
                    counter = [0,0,0,0,0,0,0,0]
                    
                    value = predicted_character
                    ser.write(value.encode('ascii'))
                    print(predicted_character)
                    cv2.putText(frame, predicted_character, (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA) #draws character on the frame
                
                elif pred_val == 7:
                    old_note = 'z'    
                    
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
                
    except:
        print("error")
        pass
        
    finally:
        pass



print("counter: " + str(counter))
ser.close()
pca.deinit()
picam2.stop()
cv2.destroyAllWindows()

