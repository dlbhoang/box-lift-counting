from ultralytics import YOLO
import cv2
import numpy as np

# ======================
# LOAD MODEL & VIDEO
# ======================
model = YOLO("0809.pt")
cap = cv2.VideoCapture("video_27.mp4")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("demo_output.mp4", fourcc, fps, (width, height))

# ======================
# ROI VÙNG XE (INITIAL)
# ======================
# ROI sẽ được cập nhật động dựa vào vị trí của xe
TRUCK_ROI_INITIAL = np.array([
    (320, 230),
    (880, 230),
    (950, 580),
    (300, 580)
], dtype=np.int32)
truck_roi = TRUCK_ROI_INITIAL.copy()

# ======================
# PARAMETERS
# ======================
MIN_CONSECUTIVE_UPWARD = 4      # phải di chuyển lên ít nhất 4 frame LIÊN TỤC
UPWARD_THRESHOLD = 1.2          # mỗi frame phải di chuyển lên ít nhất 1.2px
DOWNWARD_THRESHOLD = 0.8        # reset nếu di chuyển xuống > 0.8px
MIN_FRAMES_IN_ROI = 3           # phải ở ROI ít nhất 3 frame trước khi bắt đầu đếm
MIN_TOTAL_DISPLACEMENT = 5.0    # phải di chuyển lên TỔNG CỘNG ít nhất 5px
MAX_BOX_SIZE = 60000            # bỏ qua box quá to (> 60000px diện tích)

# ======================
# BIẾN ĐẾM
# ======================
counted_ids = set()         # IDs đã đếm
track_state = {}            # tracking state
total_boxes = 0
truck_x_offset = 0          # dùng để follow truck
last_truck_box = None       # box của xe từ frame trước

# ======================
# HÀM
# ======================
def point_in_roi(point, roi):
    """Check if point is inside ROI polygon"""
    return cv2.pointPolygonTest(roi, point, False) >= 0

def find_truck_in_frame(boxes, y_threshold=300):
    """
    Tìm xe trong frame - sử dụng box lớn nhất (cách đơn giản)
    """
    if len(boxes) == 0:
        return None
    
    # Lấy box có diện tích lớn nhất (có thể là xe)
    max_area = 0
    truck_box = None
    for box in boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > max_area and (y2 - y1) >= 30:  # chiều cao >= 30px
            max_area = area
            truck_box = box
    
    return truck_box

def get_dynamic_roi(truck_box, roi_initial):
    """
    Cập nhật ROI theo vị trí xe
    Nếu không detect được truck, dùng ROI cố định
    """
    if truck_box is None:
        return roi_initial
    
    tx1, ty1, tx2, ty2 = truck_box
    truck_cx = (tx1 + tx2) / 2
    
    # Tính offset từ vị trí xe so với ROI ban đầu
    roi_cx = (roi_initial[0][0] + roi_initial[1][0]) / 2
    offset = truck_cx - roi_cx
    
    # Dịch ROI theo offset
    new_roi = roi_initial.copy().astype(float)
    new_roi[:, 0] += offset  # dịch X
    
    return new_roi.astype(np.int32)

def get_dynamic_roi_from_boxes(box_centers, roi_initial):
    """
    Cập nhật ROI dựa vào trung bình vị trí của tất cả boxes
    Cách này tốt hơn vì theo dõi "khối boxes" trên xe
    """
    if len(box_centers) == 0:
        return roi_initial
    
    # Tính centroid của tất cả boxes
    avg_x = sum([x for x, y in box_centers]) / len(box_centers)
    
    # Tính offset từ centroid boxes so với ROI ban đầu
    roi_cx = (roi_initial[0][0] + roi_initial[1][0]) / 2
    offset = avg_x - roi_cx
    
    # Dịch ROI theo offset
    new_roi = roi_initial.copy().astype(float)
    new_roi[:, 0] += offset  # dịch X
    
    return new_roi.astype(np.int32)


# ======================
# MAIN LOOP
# ======================
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detect + Track
    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5,
        tracker="bytetrack.yaml"
    )

    # Detect truck nếu có boxes
    truck_box = None
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        truck_box = find_truck_in_frame(boxes)
    
    # Sử dụng ROI CỐ ĐỊNH (không dịch động) để tránh sai đếm
    truck_roi = TRUCK_ROI_INITIAL
    
    # Draw ROI
    cv2.polylines(frame, [truck_roi], True, (255, 0, 0), 2)
    
    # Draw truck box nếu detected
    if truck_box is not None:
        tx1, ty1, tx2, ty2 = map(int, truck_box)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 165, 255), 3)
        cv2.putText(frame, "TRUCK", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Initialize state
            if track_id not in track_state:
                track_state[track_id] = {
                    'last_y': cy,
                    'consecutive_upward': 0,      # số frame LIÊN TỤC di chuyển lên
                    'was_in_roi': False,
                    'frames_in_roi': 0,           # số frame đã ở trong ROI
                    'already_counted': False,     # đã đếm chưa
                    'total_upward_displacement': 0.0,  # tổng displacement lên (px)
                }

            state = track_state[track_id]
            is_in_roi = point_in_roi((cx, cy), truck_roi)  # Luôn tính is_in_roi
            
            # Nếu đã đếm rồi thì bỏ qua toàn bộ logic motion, chỉ vẽ
            if track_id in counted_ids:
                state['already_counted'] = True
                # Skip motion detection logic
            else:
                # Chưa đếm - tiếp tục kiểm tra logic
                # Check upward movement (Y decreasing)
                dy = state['last_y'] - cy  # positive = up
                
                if is_in_roi:
                    # Box trong ROI - tăng frame counter
                    state['frames_in_roi'] += 1
                    
                    # Kiểm tra xem box có quá to không (bỏ qua nếu > MAX_BOX_SIZE)
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area <= MAX_BOX_SIZE:
                        # Chỉ bắt đầu kiểm tra upward motion sau khi box ở ROI >= MIN_FRAMES_IN_ROI
                        if state['frames_in_roi'] >= MIN_FRAMES_IN_ROI:
                            # Kiểm tra: di chuyển lên đủ mạnh
                            if dy > UPWARD_THRESHOLD:
                                state['consecutive_upward'] += 1
                                state['total_upward_displacement'] += dy
                            elif dy < -DOWNWARD_THRESHOLD:
                                # Di chuyển xuống đáng kể → reset ngay
                                state['consecutive_upward'] = 0
                                state['total_upward_displacement'] = 0
                            # else: di chuyển nhẹ hoặc đứng yên, giữ nguyên counter
                        else:
                            # Chưa đủ frame trong ROI - reset
                            state['consecutive_upward'] = 0
                            state['total_upward_displacement'] = 0
                        
                        state['was_in_roi'] = True
                        
                        # Count nếu:
                        # 1. Đủ frame LIÊN TỤC di chuyển lên (>= MIN_CONSECUTIVE_UPWARD)
                        # 2. Đã ở ROI đủ lâu (>= MIN_FRAMES_IN_ROI)
                        # 3. Tổng displacement lên >= MIN_TOTAL_DISPLACEMENT
                        # 4. Chưa đếm trước đó
                        if (state['consecutive_upward'] >= MIN_CONSECUTIVE_UPWARD and 
                            state['frames_in_roi'] >= MIN_FRAMES_IN_ROI and
                            state['total_upward_displacement'] >= MIN_TOTAL_DISPLACEMENT and
                            track_id not in counted_ids):
                            total_boxes += 1
                            counted_ids.add(track_id)
                            state['already_counted'] = True
                            print(f"✅ COUNTED ID {track_id} | TOTAL = {total_boxes} | frame {frame_count} | consecutive={state['consecutive_upward']} | displacement={state['total_upward_displacement']:.1f}px | roi_frames={state['frames_in_roi']}")
                    else:
                        # Box quá to - bỏ qua
                        state['consecutive_upward'] = 0
                        state['total_upward_displacement'] = 0
                else:
                    # Box ngoài ROI - reset counter
                    state['consecutive_upward'] = 0
                    state['frames_in_roi'] = 0
                    state['was_in_roi'] = False
                    state['total_upward_displacement'] = 0
            
            # Determine color dựa trên trạng thái
            if state['already_counted'] or track_id in counted_ids:
                box_color = (0, 255, 0)  # GREEN - counted (vĩnh viễn)
            elif is_in_roi and state['consecutive_upward'] >= 1:
                box_color = (0, 255, 255)  # YELLOW - lifting (1+ frames)
            elif is_in_roi:
                box_color = (255, 165, 0)  # CYAN - in ROI but not lifting
            else:
                box_color = (0, 0, 255)  # RED - outside ROI

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Label with counter showing progress
            label = f"ID {track_id}"
            if state['already_counted']:
                label += " ✓DONE"  # Đánh dấu đã đếm
            elif is_in_roi and state['frames_in_roi'] >= 2:
                if state['consecutive_upward'] > 0:
                    label += f" ↑{state['consecutive_upward']}/{MIN_CONSECUTIVE_UPWARD}"
            
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # Update last position
            state['last_y'] = cy

    # Display total
    cv2.putText(
        frame,
        f"TOTAL BOXES: {total_boxes}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )
    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    # Legend
    cv2.putText(
        frame,
        "GREEN=Counted | YELLOW=Lifting... | CYAN=In ROI | RED=Outside",
        (30, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    cv2.imshow("BOX COUNTING", frame)
    out.write(frame)  # Write frame to video
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print(f"✅ FINAL TOTAL BOXES: {total_boxes}")
print("="*60)
