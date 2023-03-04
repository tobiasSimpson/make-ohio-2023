import cv2

#---- read the next frame from the capture device
def read_frame(cap):
    ret, frame = cap.read()

    if ret is False or frame is None:
        return None

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray_frame

#---- setup components
cap = cv2.VideoCapture(index=0)

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100000, varThreshold=50, detectShadows=False
)

#---- prime the accumulator
frame = read_frame(cap)
merged_frame = frame.astype(float)

#---- capture some frames
while True:
    frame = read_frame(cap)

    mask = background_subtractor.apply(frame, learningRate=0.01)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.accumulateWeighted(foreground, merged_frame, 0.1)

    cv2.imshow('Acccumulator', merged_frame)
    key = cv2.waitKey(1)

    # press 'q' to quit and save the current frame
    if key == ord('q') or key == ord('Q'):
        cv2.imwrite('merged.png', merged_frame)
        break