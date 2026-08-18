import cv2
import mediapipe as mp
cap=cv2.VideoCapture(0)
current_thickness=0
mp_hands= mp.solutions.hands
hands = mp_hands.Hands(
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5,
    max_num_hands=1,
    model_complexity=1 # 0 = fast/less accurate, 1 = balanced (default)
)
mp_draw=mp.solutions.drawing_utils  #to quickly draw hand connections (skeletons) on OpenCV images.
prev_x , prev_y = 0,0
canvas= None
shape_mode ="free"
anchor_x , anchor_y=0,0 #first pont where you put your finger to draw image
was_pen_down = False    #tracks pen state
missed_frames=0
MAX_MISSED_FRAMES=8 #slightly increases grace period so that it does not lose hand with little disturbance in hand
COLORS = {
    "Cyan": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "White": (255, 255, 255),
    "Eraser": (0, 0, 0)
}
#Draw visual color palette buttons on screen
color_boxes = [
    ("Cyan", (255, 255, 0), 10),
    ("Magenta", (255, 0, 255), 80),
    ("Yellow", (0, 255, 255), 150),
    ("Green", (0, 255, 0), 220),
    ("White", (255, 255, 255), 290),
    ("Eraser", (0, 0, 0), 360)]
current_color_name ="Magenta"
draw_color=COLORS[current_color_name]
thickness=8
SHAPE_BUTTONS=[
    ("free", "Free" , 450),
    ("line", "Line", 520),
    ("rect", "Rect", 590),
    ("circle", "Circ", 660)
]
        
while True:
    success , frame =cap.read()
    if not success:
        break
    frame = cv2.flip(frame,1)
    height , width , _= frame.shape     #'_' ignoring extra values
    if canvas is None:
        canvas=frame.copy()
        canvas[:]=0
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    pen_down= False     #pen in air
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]   #0 for wrist , 4 for thumb , 8 for index finger , 12 for middle
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS) 
        index_tip = hand.landmark[8]
        middle_tip = hand.landmark[12]
        index_mcp=hand.landmark[5]
        middle_mcp = hand.landmark[9]   #knucle/ joint btw finger and hand , used in this program to track pen up or down
        index_up= index_tip.y < index_mcp.y
        middle_up= middle_tip.y< middle_mcp.y       #y always less as it goes up
        
        y=int (index_tip.y* height)
        x= int(index_tip.x*width)  #change coordinates into pixels
        if y < 60 and index_up and not middle_up:
            if 10 <= x <= 70:
                current_color_name, draw_color = "Cyan", COLORS["Cyan"]
            elif 80 <= x <= 140:
                current_color_name, draw_color = "Magenta", COLORS["Magenta"]
            elif 150 <= x <= 210:
                current_color_name, draw_color = "Yellow", COLORS["Yellow"]
            elif 220 <= x <= 280:
                current_color_name, draw_color = "Green", COLORS["Green"]
            elif 290 <= x <= 350:
                current_color_name, draw_color = "White", COLORS["White"]
            elif 360 <= x <= 420:
                current_color_name, draw_color = "Eraser", COLORS["Eraser"]
            for mode, label, pos_x in SHAPE_BUTTONS:
                if pos_x <= x <= pos_x + 60:
                    shape_mode = mode
        if(current_color_name== 'Eraser'):
            current_thickness=10*thickness   #to erase faster
        else:
            current_thickness=thickness
        if index_up and not middle_up and y>=60:
            pen_down= True
            missed_frames=0
            cv2.circle(frame, (x,y) , 8,(0,255,0) , 10)
            
            if shape_mode == 'free':
                if prev_x!= 0 and prev_y!=0:
                    cv2.line(canvas, (prev_x,prev_y) ,(x,y) , draw_color, current_thickness)  #cv2.line(img, pt1, pt2, color, thickness)
                prev_x, prev_y = x, y
            else:
                if not was_pen_down:
                    anchor_x, anchor_y = x,y
                if shape_mode == "line":
                    cv2.line(frame, (anchor_x, anchor_y), (x, y), draw_color, current_thickness)
                elif shape_mode == "rect":
                    cv2.rectangle(frame, (anchor_x, anchor_y), (x, y), draw_color, current_thickness)
                elif shape_mode == "circle":
                    radius= int(((x-anchor_x)**2 + (y-anchor_y)**2 ) **0.5)     #distance formula
                    cv2.circle(frame, (anchor_x, anchor_y), radius, draw_color, current_thickness)
        else:
            #pen up
            cv2.circle(frame, (x,y) , 8 , (255,255,255),2)
            if was_pen_down and shape_mode != "free":      #frames is temporary and canvas is permanent as soon as aecond finger lifted frmaes change to canvas and store the drawing
                if shape_mode == "line":
                    cv2.line(canvas, (anchor_x, anchor_y), (x, y), draw_color, current_thickness)
                elif shape_mode == "rect":
                    cv2.rectangle(canvas, (anchor_x, anchor_y), (x, y), draw_color, current_thickness)
                elif shape_mode == "circle":
                    radius = int(((x - anchor_x) ** 2 + (y - anchor_y) ** 2) ** 0.5)
                    cv2.circle(canvas, (anchor_x, anchor_y), radius, draw_color, current_thickness)
            missed_frames+=1
            if missed_frames> MAX_MISSED_FRAMES:
                prev_x, prev_y =0,0 #hand still there
    else:
        missed_frames += 1
        if missed_frames > MAX_MISSED_FRAMES:
            prev_x, prev_y = 0, 0   #no hand

    was_pen_down=pen_down
    combined = cv2.add(frame, canvas)
    # NEW: Top header background bar
    cv2.rectangle(combined, (0, 0), (width, 60), (40, 40, 40), -1)

    # NEW: 
    
    for name , col , pos_x in color_boxes:
        if(name== current_color_name):  #check if that colour is selected or not and highlight it
            border=(0,255,0)
        else:
            border = (100,100,100)  #grey colour
        cv2.rectangle(combined, (pos_x, 10), (pos_x + 60, 50), col, -1)
        cv2.rectangle(combined, (pos_x, 10), (pos_x + 60, 50), border, 2)
    for name , shape, pos_x in SHAPE_BUTTONS:
        if (name ==shape_mode):
            button_bg=(0,255,0)
            border_col = (255, 255, 255)
        else:
            button_bg=(100,100,100)
            border_col=(150, 150, 150)
        
        cv2.rectangle(combined, (pos_x, 10), (pos_x + 60, 50), button_bg, -1)
        cv2.rectangle(combined, (pos_x, 10), (pos_x + 60, 50), border_col, 2)
        cv2.putText(combined, shape, (pos_x + 8, 38), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, f"Thickness: {current_thickness}", (pos_x+80, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  #text on top left corner tell free, line , rectangle,etc
    cv2.imshow("Finger Draw", combined)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
    elif key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("Saved as drawing.png")
    elif key == ord('1'):
        shape_mode = "free"
    elif key == ord('2'):
        shape_mode = "line"
    elif key == ord('3'):
        shape_mode = "rect"
    elif key == ord('4'):
        shape_mode = "circle"
    elif key == ord('+') or key == ord('='):
        thickness = min(25, thickness + 1)
    elif key == ord('-'):
        thickness = max(1, thickness - 1)
cap.release()
hands.close()
cv2.destroyAllWindows()