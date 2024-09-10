import cv2
import pyrealsense2 as rs
import numpy as np
import torch
from ultralytics import YOLO

class camera_view:
    def __init__(self, ui_given):
        # Realsense pipeline oluştur
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.ui = ui_given
        self.disable = True

        # YOLO modelini yükle
        self.model = YOLO('tomato-v3.pt')

    def read_cam(self):
        # Realsense pipeline'ı başlat
        self.pipeline.start(self.config)

        try:
            while not self.disable:
                # Kameradan bir kare oku
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Görüntüyü NumPy dizisine dönüştür
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Renk formatını BGR'den RGB'ye çevir
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # YOLO ile nesne algılama
                results = self.model(color_image)
                detections = results[0].boxes.data.cpu().numpy()


                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection[:6]
                    u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Derinlik değerini al
                    depth = depth_frame.get_distance(u, v)
                    if depth == 0:
                        continue

                    # İçsel kamera parametrelerini al
                    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

                    # 3D koordinatları (X, Y, Z) al
                    z, y, x = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                    x, y, z = x * 1000, y * 1000, z * 1000
                    y = + 10.977 + 177.98
                    z = - 24.115 + 199.02

                    label = self.model.names[int(cls)]
                    confidence = conf * 100  # confidence yüzde olarak gösterilecek

                    # Çıktıları ekrana çiz
                    cv2.rectangle(color_image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(color_image_rgb, f"Durum: {label}, Conf: {confidence:.2f}, X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}",
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Arayüzde görüntüyü güncelle
                self.ui.new_frame.setValue(color_image_rgb)

        finally:
            self.close_cam()

    def close_cam(self):
        # Realsense pipeline'ı durdur
        self.pipeline.stop()

if __name__ == "__main__":
    cam_obj = camera_view(ui_given=None)  # UI nesnesi burada tanımlanmalı
    cam_obj.read_cam()
