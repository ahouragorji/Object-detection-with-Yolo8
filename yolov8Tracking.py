import cv2
import os
from ultralytics import YOLO

model = YOLO('assets/yolov8n.pt')

video_path = "io/test_cat.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('io/outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

cat_frames = {}
frame_n = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker='assets/bytetrack.yml')
        annotated_frame = results[0].plot()
        frame_n +=1
        id_folders = {}
        
        for r in results:
            box = r.boxes
            print(box.xyxy)
            print(box.id)
            if box.id is not None:
                for i in range(len(box.id)):
                    x1, y1, x2, y2 = map(int, box.xyxy[i].cpu().numpy())
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(x1, y1, x2, y2)
                    img = annotated_frame[y1:y2, x1:x2]

                    object_id = int(box.id[i])
                    print(object_id)

                    if object_id not in id_folders:
                        id_folder = os.path.join(os.getcwd(), f"object_images", f"id_{object_id}")
                        os.makedirs(id_folder, exist_ok=True)
                        id_folders[object_id] = id_folder

                    else:
                        id_folder = id_folders[object_id]

                    cv2.imwrite(os.path.join(id_folder, f'{frame_n}.jpg'), img)
 
        # Display the annotated frame
        print(id_folders)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
