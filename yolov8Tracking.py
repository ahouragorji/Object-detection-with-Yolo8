import cv2
import os
import argparse
from ultralytics import YOLO

def main(video_path, open_folder=None, object_id=None, list_ids=False):
    model = YOLO('assets/yolov8n.pt')

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

      
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
         
            break

    cap.release()
    cv2.destroyAllWindows()



    # Display specific object if ID is provided
    if object_id:
        object_folder = os.path.join(os.getcwd(), f"object_images", f"id_{object_id}")
        os.system(f"explorer {object_folder}")

    if list_ids:
        directory_path = os.path.join(os.getcwd(), f"object_images")
        contents = os.listdir(directory_path)
        
        folders = [item for item in contents if os.path.isdir(os.path.join(directory_path, item))]
        print("Folders in the directory:")
        for folder in folders:
            print(folder)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cat Tracker CLI")
    parser.add_argument("-d", "--video_path", type=str, help="Path to the input video", required=True)
    parser.add_argument("-f", "--open_folder", action="store_true", help="Open the folder containing the results")
    parser.add_argument("--id", type=int, help="Specify the ID of the object to view")
    parser.add_argument("-l", "--list_ids", action="store_true", help="Show all detected ids")

    args = parser.parse_args()

    if args.video_path:
        main(args.video_path, args.open_folder, args.id, args.list_ids)
    else:
        parser.print_help()
