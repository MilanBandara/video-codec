import cv2

video = cv2.VideoCapture("D:\sem_7\Image_and_video_coding\Video_codec\\video-codec\istockphoto-1383024541-640_adpp_is.mp4")  # Replace with your video path

if not video.isOpened():
    print("Error opening video file")
    exit()
index = 0
while True:
    ret, frame = video.read()  # Read a frame
    index = index + 1
    if not ret:
        print("End of video")
        break

    cv2.imshow("Video", frame)  # Display the frame
    # if index%50 == 0:
    #     cv2.imwrite(f"frame_{index}.png", frame)

    
    if (480 < index) and (index < 500):
        cv2.imwrite(f"frame_{index}.png", frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print(index)