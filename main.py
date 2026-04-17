import cv2
import numpy as np

from ultralytics import YOLO

# Dictionary lưu lịch sử vị trí của mỗi object {object_id: [(x1,y1), (x2,y2), ...]}
trajectory_history = {}
MAX_TRAJECTORY_LENGTH = 60  # Số điểm tối đa trong trajectory

# Biến đếm xe IN/OUT
car_in = 0
car_out = 0

# Lưu vị trí trước đó của mỗi object để xác định hướng đi
previous_positions = {}  # {obj_id: (center_y, was_above)}

# Biến flash khi có xe đi qua
flash_timer = 0  # Frames còn lại để flash
FLASH_DURATION = 15  # Số frames flash

cap = cv2.VideoCapture("input-demo.mp4")
assert cap.isOpened(), "Error reading video file"

region_points = [[-2, 3375], [2152, 3368], [2152, 3719], [-2, 3705]]

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("output-demo.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Load model YOLO
model = YOLO("best.pt")

# Hàm vẽ bounding boxes cho objects
def draw_bboxes(im0, results):
    if results.boxes is None or len(results.boxes) == 0:
        return im0
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        obj_id = int(box.id.item()) if box.id is not None else None
        
        # Vẽ bbox màu xanh lá
        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Hiển thị ID nếu có
        if obj_id is not None:
            label = f"ID:{obj_id}"
            cv2.putText(im0, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return im0

# Hàm vẽ trajectory cho tất cả objects
def draw_trajectories(im0, trajectory_history):
    for obj_id, positions in trajectory_history.items():
        if len(positions) < 2:
            continue
        # Vẽ đường nối các điểm
        for i in range(1, len(positions)):
            # Độ dày và màu sắc thay đổi theo thời gian (điểm cũ mờ hơn)
            alpha = i / len(positions)  # 0 -> 1 từ cũ đến mới
            thickness = int(1 + alpha * 3)  # 1 -> 4
            color = (0, 255, 0)  # Màu xanh lá
            cv2.line(im0, positions[i-1], positions[i], color, thickness)
        # Vẽ điểm hiện tại
        if len(positions) > 0:
            cv2.circle(im0, positions[-1], 5, (0, 255, 255), -1)  # Vàng cho điểm mới nhất
    return im0

# Hàm xác định xe đang ở trên hay dưới đường đếm
def is_above_line(y, line_y1, line_y2, x):
    # Tính giá trị y của đường thẳng tại vị trí x
    if line_y2 == line_y1:
        return y < line_y1
    line_y_at_x = line_y1 + (line_y2 - line_y1) * (x - region_points[0][0]) / (region_points[1][0] - region_points[0][0])
    return y < line_y_at_x

# Hàm kiểm tra và cập nhật đếm xe IN/OUT
def check_crossing(obj_id, center_x, center_y):
    global car_in, car_out, flash_timer
    
    if obj_id not in previous_positions:
        previous_positions[obj_id] = (center_y, is_above_line(center_y, region_points[0][1], region_points[1][1], center_x))
        return
    
    prev_y, prev_above = previous_positions[obj_id]
    current_above = is_above_line(center_y, region_points[0][1], region_points[1][1], center_x)
    
    # Xe đi từ trên xuống dưới (IN)
    if prev_above and not current_above:
        car_in += 1
        flash_timer = FLASH_DURATION
        print(f"CAR IN! Total IN: {car_in}")
    
    # Xe đi từ dưới lên trên (OUT)
    elif not prev_above and current_above:
        car_out += 1
        flash_timer = FLASH_DURATION
        print(f"CAR OUT! Total OUT: {car_out}")
    
    previous_positions[obj_id] = (center_y, current_above)

# Cập nhật trajectory từ results tracking
def update_trajectory(results):
    global trajectory_history
    if results.boxes is None or len(results.boxes) == 0:
        return
    
    current_ids = set()
    for box in results.boxes:
        obj_id = int(box.id.item()) if box.id is not None else None
        if obj_id is None:
            continue
        
        current_ids.add(obj_id)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Tâm của bounding box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        if obj_id not in trajectory_history:
            trajectory_history[obj_id] = []
        trajectory_history[obj_id].append((center_x, center_y))
        
        # Kiểm tra xe đi qua đường đếm
        check_crossing(obj_id, center_x, center_y)
        
        # Giới hạn độ dài trajectory
        if len(trajectory_history[obj_id]) > MAX_TRAJECTORY_LENGTH:
            trajectory_history[obj_id].pop(0)
    
    # Xóa trajectory của objects đã biến mất
    disappeared_ids = set(trajectory_history.keys()) - current_ids
    for obj_id in disappeared_ids:
        del trajectory_history[obj_id]
        if obj_id in previous_positions:
            del previous_positions[obj_id]

# Process video
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Track objects với YOLO
    results = model.track(
        im0,
        persist=True,
        conf=0.1,
        iou=0.5,
        device="cuda:0",
        classes=0,
        tracker="botsort.yaml",
        verbose=False
    )
    
    # Cập nhật trajectory
    update_trajectory(results[0])
    
    # Vẽ bbox lên frame
    annotated_frame = draw_bboxes(im0, results[0])
    
    # Vẽ trajectory lên frame
    annotated_frame = draw_trajectories(annotated_frame, trajectory_history)
    
    # Vẽ thông tin đếm lên frame
    text = f"IN: {car_in}  OUT: {car_out}"
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4
    h, w = annotated_frame.shape[:2]
    (tw, th_text), _ = cv2.getTextSize(text, font, fs, th)
    x, y = (w - tw)//2, h//3
    # background
    cv2.rectangle(annotated_frame,
                (x-10, y-th_text-10),
                (x+tw+10, y+10),
                (0, 0, 0), -1)
    # text
    cv2.putText(annotated_frame, text, (x, y),
                font, fs, (0,255,0), th, cv2.LINE_AA)
    
    # Vẽ vùng đếm (region) - nhấp nháy khi có xe đi qua
    region_np = np.array(region_points, dtype=np.int32)
    if flash_timer > 0:
        # Nháy nền vàng khi có xe đi qua
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, [region_np], (0, 255, 255))  # Nền vàng
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        cv2.polylines(annotated_frame, [region_np], isClosed=True, color=(0, 255, 255), thickness=3)
        flash_timer -= 1
    else:
        cv2.polylines(annotated_frame, [region_np], isClosed=True, color=(255, 0, 0), thickness=2)

    frame_count += 1
    if frame_count % MAX_TRAJECTORY_LENGTH == 0:
        print(f"Processed {frame_count} frames...")

    video_writer.write(annotated_frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()