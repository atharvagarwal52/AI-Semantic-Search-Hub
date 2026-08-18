import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5,
    max_num_hands=1,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

prev_x, prev_y = 0, 0
canvas = None
shape_mode = "free"
anchor_x, anchor_y = 0, 0
was_pen_down = False
missed_frames = 0
MAX_MISSED_FRAMES = 8

COLORS = {
    "Cyan": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "White": (255, 255, 255),
    "Eraser": (0, 0, 0)
}
current_color_name = "Magenta"
draw_color = COLORS[current_color_name]
thickness = 8

# NEW: Define shape buttons layout (Name, Keyboard Shortcut / Label, Start X Position)
SHAPE_BUTTONS = [
    ("free", "Free", 450),
    ("line", "Line", 520),
    ("rect", "Rect", 590),
    ("circle", "Circ", 660)
]

while True:
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    if canvas is None:
        canvas = frame.copy()
        canvas[:] = 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    pen_down = False

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        
        index_tip = hand.landmark[8]
        middle_tip = hand.landmark[12]
        index_mcp = hand.landmark[5]
        middle_mcp = hand.landmark[9]
        
        index_up = index_tip.y < index_mcp.y
        middle_up = middle_tip.y < middle_mcp.y
        
        y = int(index_tip.y * height)
        x = int(index_tip.x * width)
        
        # --- UI HEADER INTERACTION (y < 60) ---
        if y < 60 and index_up and not middle_up:
            # Color Selection
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
            
            # NEW: Shape Selection via UI Header
            for mode, label, pos_x in SHAPE_BUTTONS:
                if pos_x <= x <= pos_x + 60:
                    shape_mode = mode

        if current_color_name == 'Eraser':
            current_thickness = 10 * thickness
        else:
            current_thickness = thickness

        # --- DRAWING AREA INTERACTION (y >= 60) ---
        if index_up and not middle_up and y >= 60:
            pen_down = True
            missed_frames = 0
            cv2.circle(frame, (x, y), 8, (0, 255, 0), 10)
            
            if shape_mode == 'free':
                if prev_x != 0 and prev_y != 0:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, current_thickness)
                prev_x, prev_y = x, y
            else:
                if not was_pen_down:
                    anchor_x, anchor_y = x, y
                if shape_mode == "line":
                    cv2.line(frame, (anchor_x, anchor_y), (x, y), draw_color, current_thickness)
                elif shape_mode == "rect":
                    cv2.rectangle(frame, (anchor_x, anchor_y), (x, y), draw_color, current_thickness)
                elif shape_mode == "circle":
                    radius = int(((x - anchor_x)**2 + (y - anchor_y)**2)**0.5)
                    cv2.circle(frame, (anchor_x, anchor_y), radius, draw_color, current_thickness)
        else:
            # Pen up
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
            if was_pen_down and shape_mode != "free":
                if shape_mode == "line":
                    cv2.line(canvas, (anchor_x, anchor_y), (x, y), draw_color, current_thickness)
                elif shape_mode == "rect":
                    cv2.rectangle(canvas, (anchor_x, anchor_y), (x, y), draw_color, current_thickness)
                elif shape_mode == "circle":
                    radius = int(((x - anchor_x)**2 + (y - anchor_y)**2)**0.5)
                    cv2.circle(canvas, (anchor_x, anchor_y), radius, draw_color, current_thickness)
            missed_frames += 1
            if missed_frames > MAX_MISSED_FRAMES:
                prev_x, prev_y = 0, 0
    else:
        missed_frames += 1
        if missed_frames > MAX_MISSED_FRAMES:
            prev_x, prev_y = 0, 0

    was_pen_down = pen_down
    combined = cv2.add(frame, canvas)

    # Header Background Bar
    cv2.rectangle(combined, (0, 0), (width, 60), (40, 40, 40), -1)

    # Render Color Buttons
    color_boxes = [
        ("Cyan", (255, 255, 0), 10),
        ("Magenta", (255, 0, 255), 80),
        ("Yellow", (0, 255, 255), 150),
        ("Green", (0, 255, 0), 220),
        ("White", (255, 255, 255), 290),
        ("Eraser", (0, 0, 0), 360)
    ]
    for name, col, pos_x in color_boxes:
        border = (0, 255, 0) if name == current_color_name else (100, 100, 100)
        cv2.rectangle(combined, (pos_x, 10), (pos_x + 60, 50), col, -1)
        cv2.rectangle(combined, (pos_x, 10), (pos_x + 60, 50), border, 2)

    # NEW: Render Visual Shape Selection Buttons
    for mode, label, pos_x in SHAPE_BUTTONS:
        is_active = (mode == shape_mode)
        btn_bg = (0, 200, 0) if is_active else (80, 80, 80)
        border_col = (255, 255, 255) if is_active else (150, 150, 150)
        
        cv2.rectangle(combined, (pos_x, 10), (pos_x + 60, 50), btn_bg, -1)
        cv2.rectangle(combined, (pos_x, 10), (pos_x + 60, 50), border_col, 2)
        cv2.putText(combined, label, (pos_x + 8, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display Brush Size Indicator
    cv2.putText(combined, f"Size: {thickness}", (width - 120, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

# NEW: Clean resource cleanup
cap.release()
hands.close()
cv2.destroyAllWindows()