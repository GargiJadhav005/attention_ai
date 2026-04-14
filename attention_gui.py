import cv2
import mediapipe as mp
import time
import csv
import matplotlib.pyplot as plt
import threading
import tkinter as tk

# ========== MediaPipe Setup ==========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_eye_ratio(landmarks, eye_indices, w, h):
    points = []
    for i in eye_indices:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        points.append((x, y))

    vertical = abs(points[1][1] - points[5][1])
    horizontal = abs(points[0][0] - points[3][0])

    return vertical / horizontal if horizontal != 0 else 0

# ========== Global Variables ==========
running = False

def start_detection():
    global running
    running = True
    threading.Thread(target=run_detection).start()

def stop_detection():
    global running
    running = False

# ========== Main Detection ==========
def run_detection():
    global running

    cap = cv2.VideoCapture(0)

    start_time = time.time()
    attentive_time = 0

    last_status = "ATTENTIVE"
    distraction_start = None
    distracted_periods = []

    time_list = []
    attention_list = []
    data = []

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "NOT ATTENTIVE"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left = get_eye_ratio(landmarks, LEFT_EYE, w, h)
                right = get_eye_ratio(landmarks, RIGHT_EYE, w, h)
                avg = (left + right) / 2

                if avg > 0.2:
                    status = "ATTENTIVE"
                else:
                    status = "DISTRACTED"

        current_time = time.time()

        # Attention tracking
        if status == "ATTENTIVE":
            attentive_time += 1

            if last_status == "DISTRACTED" and distraction_start:
                distracted_periods.append((distraction_start, current_time))
                distraction_start = None
        else:
            if last_status == "ATTENTIVE":
                distraction_start = current_time

        last_status = status

        total_time = current_time - start_time
        attention_score = (attentive_time / total_time) * 100 if total_time > 0 else 0

        t = current_time - start_time
        time_list.append(t)
        attention_list.append(attention_score)
        data.append([t, attention_score, status])

        # Countdown message
        remaining = max(0, 21 - int(t))
        msg = f"Closing in {remaining}s" if remaining > 0 else "Stopping..."

        # Display
        cv2.putText(frame, f"Status: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Score: {int(attention_score)}%", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.putText(frame, msg, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Attention Monitor", frame)

        # Auto stop after 21 sec
        if t >= 21:
            running = False

        if cv2.waitKey(1) & 0xFF == 27:
            running = False

    cap.release()
    cv2.destroyAllWindows()

    # Save CSV
    with open("attention_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Score", "Status"])
        writer.writerows(data)

    print("\n=== SESSION REPORT ===")
    print(f"Total Time: {round(total_time,2)} sec")
    print(f"Attention Score: {round(attention_score,2)} %")

    print("\nDistraction Periods:")
    for start, end in distracted_periods:
        print(f"From {round(start - start_time,2)}s to {round(end - start_time,2)}s")

    # Graph
    plt.plot(time_list, attention_list)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Attention Score")
    plt.title("Attention vs Time")
    plt.show()

# ========== GUI ==========
root = tk.Tk()
root.title("Attention Monitoring System")
root.geometry("300x200")

start_btn = tk.Button(root, text="Start", command=start_detection, bg="green", fg="white")
start_btn.pack(pady=20)

stop_btn = tk.Button(root, text="Stop", command=stop_detection, bg="red", fg="white")
stop_btn.pack(pady=20)

root.mainloop()