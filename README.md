# Đếm Xe Với YOLO Tracking

## Giới Thiệu

Project đếm xe ra/vào sử dụng YOLO tracking kết hợp với trajectory visualization.

## Tính Năng

- **Tracking đối tượng**: Theo dõi xe bằng YOLO và BotSort tracker
- **Đếm IN/OUT**: Đếm số xe đi vào và đi ra qua vùng đếm (region)
- **Trajectory**: Hiển thị đường đi của mỗi xe
- **Visual feedback**: Vùng đếm nhấp nháy vàng khi có xe đi qua
- **Bounding boxes**: Hiển thị khung xe kèm ID

## Video Demo

![Demo Video](/d:/Projects_Personal/Project_Car_Counting/output-demo.avi)

## Cách Sử Dụng

```bash
python main.py
```

## Cấu Hình

| Tham Số | Mô Tả |
|---------|--------|
| `model` | Đường dẫn model YOLO (.pt) |
| `region_points` | Tọa độ vùng đếm xe |
| `classes` | Lớp object cần detection |
| `MAX_TRAJECTORY_LENGTH` | Độ dài tối đa của trajectory |

## Yêu Cầu

- Python 3.8+
- ultralytics
- OpenCV
- NumPy
- CUDA (khuyến nghị)