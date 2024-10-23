import cv2
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# TrackerState class holds the state transition matrix (F) and measurement matrix (H)
class TrackerState(torch.nn.Module):
    def __init__(self, dt, device="cpu"):
        super(TrackerState, self).__init__()
        self.dt = dt
        # State transition matrix for position and velocity
        self.F = torch.tensor([
            [1, 0, self.dt, 0, 0, 0],  # x
            [0, 1, 0, self.dt, 0, 0],  # y
            [0, 0, 1, 0, 0, 0],        # vx
            [0, 0, 0, 1, 0, 0],        # vy
            [0, 0, 0, 0, 1, 0],        # w
            [0, 0, 0, 0, 0, 1]         # h
        ], dtype=torch.float32, device=device)

        # Measurement matrix for detecting position
        self.H = torch.tensor([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 0, 0, 1, 0],  # w
            [0, 0, 0, 0, 0, 1]   # h
        ], dtype=torch.float32, device=device)

# Tracker class for each detected object using Kalman filter for state prediction and update
class Tracker(torch.nn.Module):
    def __init__(self, id, box, dt, device="cpu"):
        super(Tracker, self).__init__()
        self.id = id
        self.dt = dt
        self.state = TrackerState(dt, device=device)
        # State vector: [center_x, center_y, velocity_x, velocity_y, width, height]
        self.x_hat = torch.tensor([box[0] + box[2] / 2, box[1] + box[3] / 2, 0, 0, box[2], box[3]], dtype=torch.float32, device=device)
        self.P = torch.eye(6, dtype=torch.float32, device=device) * 1000.0  # Initial covariance matrix
        self.Q = torch.eye(6, dtype=torch.float32, device=device) * 0.1    # Process noise covariance
        self.R = torch.eye(4, dtype=torch.float32, device=device) * 1.0    # Measurement noise covariance

    # Predict the next state based on the current state and process model
    def predict(self):
        self.x_hat = torch.matmul(self.state.F, self.x_hat)  # State prediction
        self.P = torch.matmul(torch.matmul(self.state.F, self.P), self.state.F.T) + self.Q  # Covariance prediction

    # Update the state based on the measurement
    def update(self, z):
        S = torch.matmul(torch.matmul(self.state.H, self.P), self.state.H.T) + self.R  # Innovation covariance
        K = torch.matmul(torch.matmul(self.P, self.state.H.T), torch.linalg.inv(S))    # Kalman gain
        y = z - torch.matmul(self.state.H, self.x_hat)  # Innovation or measurement residual
        self.x_hat += torch.matmul(K, y)  # State update
        self.P -= torch.matmul(K, torch.matmul(self.state.H, self.P))  # Covariance update

    # Get the current state
    def get_state(self):
        return self.x_hat.clone()

    # Get the bounding box from the state vector
    def get_bounding_box(self):
        x, y, w, h = self.x_hat[0], self.x_hat[1], self.x_hat[4], self.x_hat[5]
        return [self.id, int(x - w / 2), int(y - h / 2), int(w), int(h)]

# MultiObjectTracker manages multiple Tracker objects
class MultiObjectTracker:
    def __init__(self, dt, device="cpu"):
        self.trackers = []  # List of trackers
        self.next_id = 0    # ID for the next tracker
        self.dt = dt
        self.device = device

    # Update the trackers with new detections
    def update(self, detections):
        detections_tensor = torch.tensor([[det[0] + det[2] / 2, det[1] + det[3] / 2, det[2], det[3]] for det in detections], dtype=torch.float32, device=self.device)
        
        if len(detections_tensor) == 0:
            return

        if not self.trackers:
            for det in detections:
                self.trackers.append(Tracker(self.next_id, det, self.dt, device=self.device))  # Create new tracker
                self.next_id += 1
        else:
            for tracker in self.trackers:
                tracker.predict()  # Predict the next state for each tracker

            predicted_positions = torch.stack([tracker.get_state()[:4] for tracker in self.trackers])
            
            if len(predicted_positions) == 0 or len(detections_tensor) == 0:
                return

            cost_matrix = torch.cdist(predicted_positions[:, :2], detections_tensor[:, :2])  # Compute cost matrix

            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())  # Solve assignment problem
            unmatched_trackers = set(range(len(self.trackers))) - set(row_ind)
            unmatched_detections = set(range(len(detections))) - set(col_ind)

            for r, c in zip(row_ind, col_ind):
                self.trackers[r].update(detections_tensor[c])  # Update matched trackers
            for u in unmatched_detections:
                self.trackers.append(Tracker(self.next_id, detections[u], self.dt, device=self.device))  # Create new trackers for unmatched detections
                self.next_id += 1
            self.trackers = [self.trackers[i] for i in range(len(self.trackers)) if i not in unmatched_trackers]  # Remove unmatched trackers

    # Get bounding boxes of all trackers
    def get_bounding_boxes(self):
        return [tracker.get_bounding_box() for tracker in self.trackers]

# Load YOLOv8 model
def load_yolov8_model():
    model = YOLO('yolov8n.pt')  # Load YOLOv8 model
    return model

# Detect objects in the frame using YOLOv8
def detect_objects(frame, model):
    results = model(frame)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            boxes.append([x1, y1, w, h])
    return boxes

# Calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def main():
    model = load_yolov8_model()  # Load YOLOv8 model
    cap = cv2.VideoCapture(r"C:\Users\ATIK\video.mp4")  # Open video file
    mot = MultiObjectTracker(1 / 30, device="cuda")  # Initialize multi-object tracker

    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break
        detections = detect_objects(frame, model)  # Detect objects in the frame
        detections_tensor = torch.tensor(detections, dtype=torch.float32, device=mot.device)  # Convert detections to tensor
        mot.update(detections)  # Update the trackers with new detections
        for bbox in mot.get_bounding_boxes():
            id, x, y, w, h = bbox
            iou = 0.0
            if len(detections) > 0:
                iou = max(calculate_iou(detections_tensor[i].cpu().numpy(), [x, y, w, h]) for i in range(len(detections)))  # Calculate IoU for each detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, f'Score: {iou:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display IoU
        cv2.imshow('Frame', frame)  # Show the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
