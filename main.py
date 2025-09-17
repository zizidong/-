import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import time
import serial
from serial.tools import list_ports


def _is_rotation_matrix(R: np.ndarray) -> bool:
    Rt = np.transpose(R)
    should_be_identity = Rt @ R
    I = np.identity(3, dtype=R.dtype)
    return np.linalg.norm(I - should_be_identity) < 1e-6


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
    assert _is_rotation_matrix(R), "Invalid rotation matrix"
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def _landmark_to_points(landmarks, image_shape):
    h, w = image_shape[:2]
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts


def estimate_head_pose(image, face_landmarks) -> tuple:
    h, w = image.shape[:2]
    # 3D model points (approx) for selected facial landmarks (in mm)
    model_points = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -63.6, -12.5],      # Chin
        [-43.3, 32.7, -26.0],     # Left eye left corner
        [43.3, 32.7, -26.0],      # Right eye right corner
        [-28.9, -28.9, -24.1],    # Left Mouth corner
        [28.9, -28.9, -24.1],     # Right mouth corner
    ], dtype=np.float64)

    # MediaPipe FaceMesh landmark indices corresponding to the above points
    # Nose tip (1), Chin (152), Left eye left corner (33), Right eye right corner (263),
    # Left mouth corner (61), Right mouth corner (291)
    idxs = [1, 152, 33, 263, 61, 291]

    image_points = []
    for idx in idxs:
        lm = face_landmarks.landmark[idx]
        image_points.append([lm.x * w, lm.y * h])
    image_points = np.array(image_points, dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    eulers = _rotation_matrix_to_euler_angles(R)  # radians (x: pitch, y: yaw, z: roll)
    pitch, yaw, roll = np.degrees(eulers[0]), np.degrees(eulers[1]), np.degrees(eulers[2])
    return float(yaw), float(pitch), float(roll)


class HeadPoseApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Nose Tracker (YOLOv8 Pose)")
        # 표시 영상 너비(픽셀) - 창 크기를 작게 유지
        self.display_width = 640

        # 출력 파일 준비
        safe_dir = Path.home() / "facepose_outputs"
        safe_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = safe_dir / "face_angles.txt"
        with self.output_path.open("w", encoding="utf-8") as f:
            f.write("timestamp\tyaw(deg)\tpitch(deg)\troll(deg)\n")

        # 캡처/모델 초기화
        self.cap = self._open_capture()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다.")
        # YOLOv8 Pose 모델 (자동 다운로드)
        self.yolo = YOLO("yolov8n-pose.pt")

        # 코 좌표 로그 파일
        self.nose_log_path = safe_dir / "nose_positions.txt"
        with self.nose_log_path.open("w", encoding="utf-8") as f:
            f.write("timestamp\tx\ty\tyaw(deg)\tpitch(deg)\n")

        # 직렬 통신(아두이노) 초기화
        self.serial = self._open_serial()

        # UI 구성
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.angles_var = tk.StringVar(value="Nose: (-, -)  Yaw: -  Pitch: -")
        self.angles_label = tk.Label(self.root, textvariable=self.angles_var, font=("Segoe UI", 12))
        self.angles_label.pack(pady=6)

        # 직렬 상태 라벨
        self.serial_var = tk.StringVar(value=f"Serial: {'CONNECTED' if (self.serial and getattr(self.serial, 'is_open', False)) else 'DISCONNECTED'}")
        self.serial_label = tk.Label(self.root, textvariable=self.serial_var, font=("Segoe UI", 10))
        self.serial_label.pack(pady=2)

        # COM 포트 선택/연결 UI
        ports_frame = tk.Frame(self.root)
        ports_frame.pack(fill=tk.X, padx=0, pady=2)
        tk.Label(ports_frame, text="Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value="")
        self.port_combo = ttk.Combobox(ports_frame, textvariable=self.port_var, width=18, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=4)
        self.refresh_btn = tk.Button(ports_frame, text="Refresh", command=self.on_refresh_ports)
        self.refresh_btn.pack(side=tk.LEFT, padx=2)
        self.connect_btn = tk.Button(ports_frame, text="Connect", command=self.on_connect_click)
        self.connect_btn.pack(side=tk.LEFT, padx=2)
        self.disconnect_btn = tk.Button(ports_frame, text="Disconnect", command=self.on_disconnect_click)
        self.disconnect_btn.pack(side=tk.LEFT, padx=2)
        self.test_btn = tk.Button(ports_frame, text="Test Send", command=self.on_test_send)
        self.test_btn.pack(side=tk.LEFT, padx=6)

        # 직렬 모니터 영역
        monitor_frame = tk.Frame(self.root)
        monitor_frame.pack(fill=tk.BOTH, expand=False, pady=4)
        tk.Label(monitor_frame, text="Serial Monitor", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.serial_text = tk.Text(monitor_frame, height=8, width=90)
        self.serial_text.pack(fill=tk.BOTH, expand=False)
        self.serial_text.configure(state=tk.DISABLED)

        # 종료 처리 연결
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 주기 업데이트 시작
        self.update_frame()
        self.poll_serial()
        self.on_refresh_ports()

    def _open_capture(self):
        # Windows에서 MSMF 이슈가 있을 수 있어 DirectShow 우선 시도
        candidates = [
            (0, cv2.CAP_DSHOW),
            (0, cv2.CAP_MSMF),
            (0, 0),
            (1, cv2.CAP_DSHOW),
            (1, 0),
            (2, cv2.CAP_DSHOW),
        ]
        for index, backend in candidates:
            try:
                cap = cv2.VideoCapture(index, backend) if backend != 0 else cv2.VideoCapture(index)
                if cap is not None and cap.isOpened():
                    # 선택적으로 해상도 설정(실패해도 무시)
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    except Exception:
                        pass
                    return cap
                if cap is not None:
                    cap.release()
            except Exception:
                continue
        return None

    def _open_serial(self):
        # 우선 환경변수 ARDUINO_COM 또는 기본 후보(COM3~COM10) 시도, Arduino 문자열 우선 선택
        try_ports = []
        env_port = os.environ.get("ARDUINO_COM")
        if env_port:
            try_ports.append(env_port)
        detected = list(list_ports.comports())
        arduino_ports = [p.device for p in detected if "arduino" in (p.description or "").lower()]
        try_ports.extend(arduino_ports)
        if not try_ports:
            try_ports.extend([f"COM{i}" for i in range(3, 11)])
        for port in try_ports:
            try:
                ser = serial.Serial(port=port, baudrate=115200, timeout=0.1)
                # 아두이노 자동 리셋 대기 및 버퍼 비우기
                time.sleep(2.0)
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass
                self.port_var.set(port)
                return ser
            except Exception:
                continue
        return None

    def _list_serial_ports(self):
        items = []
        try:
            for p in list_ports.comports():
                label = f"{p.device} - {p.description}"
                items.append((p.device, label))
        except Exception:
            pass
        return items

    def on_refresh_ports(self):
        items = self._list_serial_ports()
        labels = [lbl for _, lbl in items]
        self.port_combo["values"] = labels
        # 선택 유지/초기화
        if labels:
            current = self.port_var.get()
            if not current or all(current.split(" ")[0] != dev for dev, _ in items):
                self.port_var.set(items[0][0])
        else:
            self.port_var.set("")

    def on_connect_click(self):
        # 다른 프로그램(IDE 시리얼 모니터)이 포트를 점유하면 실패함
        sel = self.port_var.get()
        port = sel.split(" ")[0] if sel else sel
        if not port:
            self._append_serial_log('ERR', 'No port selected')
            return
        try:
            if self.serial and getattr(self.serial, 'is_open', False):
                self.serial.close()
        except Exception:
            pass
        try:
            self.serial = serial.Serial(port=port, baudrate=115200, timeout=0.1)
            time.sleep(2.0)
            try:
                self.serial.reset_input_buffer()
                self.serial.reset_output_buffer()
            except Exception:
                pass
            self.serial_var.set("Serial: CONNECTED")
            self._append_serial_log('INFO', f'Connected {port}')
        except Exception as e:
            self.serial = None
            self.serial_var.set("Serial: DISCONNECTED")
            self._append_serial_log('ERR', f'Connect failed: {e}')

    def on_disconnect_click(self):
        try:
            if self.serial and getattr(self.serial, 'is_open', False):
                port = self.serial.port
                self.serial.close()
                self._append_serial_log('INFO', f'Disconnected {port}')
        except Exception:
            pass
        self.serial_var.set("Serial: DISCONNECTED")

    def on_test_send(self):
        if self.serial and getattr(self.serial, 'is_open', False):
            try:
                msg = "PING\r\n"
                self.serial.write(msg.encode('utf-8'))
                self.serial.flush()
                self._append_serial_log('TX', msg.strip())
            except Exception as e:
                self._append_serial_log('ERR', f'TX failed: {e}')
        else:
            self._append_serial_log('ERR', 'Not connected')

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # 읽기 실패 시 캡처 재초기화 재시도
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            self.cap = self._open_capture()
            self.root.after(200, self.update_frame)
            return

        # YOLOv8 Pose 추론 (COCO 키포인트: 0 = nose)
        results = self.yolo.predict(frame, verbose=False)
        if results and len(results) > 0:
            kp = results[0].keypoints
            if kp is not None and kp.xy is not None and len(kp.xy) > 0:
                # 가장 첫 번째 사람의 코 좌표
                pts = kp.xy[0].tolist()  # shape: (17, 2)
                if len(pts) >= 1:
                    nose_x, nose_y = pts[0]
                    h, w = frame.shape[:2]
                    # 각도 근사: 화면 중심 대비 비율 * 화각/2
                    hfov_deg = 60.0
                    vfov_deg = 35.0
                    yaw = ((nose_x - (w / 2)) / (w / 2)) * (hfov_deg / 2)
                    pitch = -((nose_y - (h / 2)) / (h / 2)) * (vfov_deg / 2)
                    # 시각화
                    cv2.circle(frame, (int(nose_x), int(nose_y)), 6, (0, 0, 255), -1)
                    cv2.putText(frame, "Nose", (int(nose_x)+8, int(nose_y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # 라벨 및 로그
                    self.angles_var.set(f"Nose: ({nose_x:.1f}, {nose_y:.1f})  Yaw: {yaw:.1f}  Pitch: {pitch:.1f}")
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    with self.nose_log_path.open("a", encoding="utf-8") as f:
                        f.write(f"{ts}\t{nose_x:.2f}\t{nose_y:.2f}\t{yaw:.2f}\t{pitch:.2f}\n")
                    # 아두이노로 전송
                    if self.serial is not None and getattr(self.serial, 'is_open', False):
                        try:
                            msg = f"NOSE,{nose_x:.1f},{nose_y:.1f},{yaw:.1f},{pitch:.1f}\r\n"
                            self.serial.write(msg.encode("utf-8"))
                            self.serial.flush()
                            # 상태 표시 업데이트(연결 확인)
                            if self.serial_var.get() != "Serial: CONNECTED":
                                self.serial_var.set("Serial: CONNECTED")
                            self._append_serial_log('TX', msg.strip())
                        except Exception:
                            pass

        # Tkinter로 표시
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # GUI 표시용 리사이즈 (가로 고정, 세로 비율 유지)
        try:
            w, h = img.size
            if w > 0 and self.display_width > 0:
                target_w = self.display_width
                target_h = int(h * (target_w / w))
                if target_w > 0 and target_h > 0:
                    img = img.resize((target_w, target_h), Image.BILINEAR)
        except Exception:
            pass
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_close(self):
        # YOLO는 별도 close 불필요
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            if self.serial is not None:
                self.serial.close()
        except Exception:
            pass
        self.root.destroy()

    def _append_serial_log(self, direction: str, text: str):
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {direction}: {text}\n"
            self.serial_text.configure(state=tk.NORMAL)
            self.serial_text.insert(tk.END, line)
            self.serial_text.see(tk.END)
            self.serial_text.configure(state=tk.DISABLED)
        except Exception:
            pass

    def poll_serial(self):
        try:
            if self.serial is not None and getattr(self.serial, 'is_open', False):
                bytes_waiting = getattr(self.serial, 'in_waiting', 0)
                read_guard = 0
                while bytes_waiting and read_guard < 50:  # 한번에 과도하게 읽지 않도록 제한
                    try:
                        raw = self.serial.readline()
                        if not raw:
                            break
                        msg = raw.decode('utf-8', errors='ignore').strip()
                        if msg:
                            self._append_serial_log('RX', msg)
                    except Exception:
                        break
                    read_guard += 1
                    bytes_waiting = getattr(self.serial, 'in_waiting', 0)
        except Exception:
            pass
        # 50ms 주기 폴링
        self.root.after(50, self.poll_serial)


def main():
    root = tk.Tk()
    app = HeadPoseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


