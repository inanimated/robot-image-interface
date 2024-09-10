import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import serial


# ser = serial.Serial('COM10', 115200)  # Port ve baud rate'i uygun şekilde ayarlayın

# Load the YOLO model
model = YOLO('tomato-v3.pt')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5)  # 5 is short range

thetas = [0, 0, 0, 0, 0, 0]

x, y, z = None, None, None

def commands(command):  # Robotun ana konum theta değerleri

    if command ==  'home':
        thetas = [0, 0, 0, 0, 0, 0]
        return thetas
    elif command == 'photo':
        thetas = [0, -30, 38, 0, 0, 0]
        return thetas
    elif command == 'catch':
        thetas = [0, 28, -20, 0, 0, 0]
        return thetas
    elif command == 'final':
        thetas = [-60, -25, 40, 45, 45, 0]
        return thetas
    else:
        return "Invalid!"

def process_angle(angle):
    if angle <= 255:
        return [0, angle]
    else:
        high_byte = angle // 256
        low_byte = angle % 256
        return [high_byte, low_byte]

def send_angles_via_uart(angles):
    bit_stream = bytearray()
    for angle in thetas:
        bit_stream.extend(process_angle(angle))
    #ser.write(bit_stream)
    print(bit_stream)

def Cam2Robot(y, z): # Kameradan alınan eksenleri robota göre düzenleme
    y = y + 10.977 + 177.98
    z = z - 24.115 + 199.02
    return y, z

##################KINEMATICS#############

# Dereceleri radyana dönüştürme
def deg_to_rad(deg):
    return deg * np.pi / 180

# Radyanları dereceye dönüştürme
def rad_to_deg(rad):
    return rad * 180 / np.pi

# XYZ'den Tform matrisini oluştur
def create_tform(x, y, z):
    # Sabit yönelim matrisi
    rotation_matrix = np.array([
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0]
    ])
    tform = np.vstack((np.hstack((rotation_matrix, [[x], [y], [z]])), [0, 0, 0, 1]))
    return tform

# Ters kinematik hesaplama fonksiyonu
def inverse_kinematics(tform):
    pos = tform[:3, 3]
    rotMat = tform[:3, :3]

    L2 = 246
    a = 28
    L3 = 200
    L4 = 100
    WCP_pos = pos - rotMat @ np.array([0, 0, L4])
    px, py, pz = WCP_pos

    d = (px ** 2 + py ** 2 + pz ** 2 - L2 ** 2 - a ** 2 - L3 ** 2) / (2 * L2)
    temp1 = a ** 2 + L3 ** 2 - d ** 2

    if temp1 < 0:
        return []  # Geçerli çözüm yok

    th3 = [np.arctan2(-L3, a) + np.arctan2(np.sqrt(temp1), d),
           np.arctan2(-L3, a) + np.arctan2(-np.sqrt(temp1), d)]

    solutions = []
    for theta3 in th3:
        e = -a * np.sin(theta3) - L3 * np.cos(theta3)
        f = L2 + a * np.cos(theta3) - L3 * np.sin(theta3)
        temp2 = e ** 2 + f ** 2 - pz ** 2

        if temp2 >= 0:
            th2 = [np.arctan2(e, f) + np.arctan2(np.sqrt(temp2), pz),
                   np.arctan2(e, f) + np.arctan2(-np.sqrt(temp2), pz)]
            for theta2 in th2:
                g = L3 * np.cos(theta2 + theta3) + a * np.sin(theta2 + theta3) + L2 * np.sin(theta2)
                th1 = np.arctan2(py * g, px * g)
                solutions.append([th1, theta2, theta3])

    if not solutions:
        return []

    thetaOut = []
    for th1, th2, th3 in solutions:
        R0w = np.array([
            [np.sin(th2 + th3) * np.cos(th1), np.sin(th1), np.cos(th2 + th3) * np.cos(th1)],
            [np.sin(th2 + th3) * np.sin(th1), -np.cos(th1), np.cos(th2 + th3) * np.sin(th1)],
            [np.cos(th2 + th3), 0, -np.sin(th2 + th3)]
        ])
        Rw6 = np.linalg.inv(R0w) @ rotMat

        rr = np.sqrt(Rw6[0, 2] ** 2 + Rw6[1, 2] ** 2)
        th5 = [np.arctan2(rr, Rw6[2, 2]), np.arctan2(-rr, Rw6[2, 2])]

        for theta5 in th5:
            if theta5 == 0:
                th4 = np.arctan2(Rw6[1, 0], Rw6[0, 0])
                th6 = 0
            elif theta5 == np.pi:
                th4 = np.arctan2(-Rw6[1, 0], -Rw6[0, 0])
                th6 = 0
            else:
                th4 = np.arctan2(-Rw6[1, 2] / rr, -Rw6[0, 2] / rr)
                th6 = np.arctan2(-Rw6[2, 1] / rr, Rw6[2, 0] / rr)

            thetaOut.append([th1, th2, th3, th4, theta5, th6])

    return thetaOut

# Eklem limitlerini kontrol etme
def apply_joint_limits(theta_out, joint_limits):
    min_limits = np.deg2rad(joint_limits[0])
    max_limits = np.deg2rad(joint_limits[1])

    valid_solutions = []
    for angles in theta_out:
        valid = True
        for i, angle in enumerate(angles):
            if not (min_limits[i] <= angle <= max_limits[i]):
                valid = False
                break
        if valid:
            valid_solutions.append(angles)
    return valid_solutions

# XYZ ile ters kinematik hesaplama fonksiyonu
def inverse_kinematics_xyz(x, y, z):
    tform = create_tform(x, y, z)
    result = inverse_kinematics(tform)

    # Eklem limitlerini uygulayın
    joint_limits = [
        [-140, -60, -135, -180, -45, -180],  # Minimum limitler
        [140, 130, 70, 180, 90, 180]  # Maksimum limitler
    ]
    valid_result = apply_joint_limits(result, joint_limits)

    return valid_result

# Hesaplama fonksiyonu
def calculate_ik(x, y, z):
    try:
        result = inverse_kinematics_xyz(x, y, z)

        if not result:
            return 0
        else:
            result_array = [[rad_to_deg(angle) for angle in angles] for angles in result]
            print("Ters kinematik sonuçları:")
            for angles in result_array:
                print(angles)
    except Exception as e:
        return "0 {e}"

##################KINEMATICS#############


def main_loop():
    global x, y, z

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image)

            detections = results[0].boxes.data.cpu().numpy()  # YOLOv8 sonuçları

            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection[:6]
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Derinlik değerini al
                depth = depth_frame.get_distance(u, v)
                if depth == 0:  # Geçersiz derinlik değerlerini atla
                    continue

                # İçsel kamera parametrelerini al
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

                # 3D koordinatları (X, Y, Z) al
                z, y, x = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                x = x * 1000
                y = y * 1000
                z = z * 1000
                y, z = Cam2Robot(y, z)

                print(f"Object {int(cls)}: X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}")



                # Çıktıları ekrana çiz
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(color_image, f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Görüntüleri göster
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)

            # 's' tuşuna basıldığında hesaplama yapılır
            if cv2.waitKey(1) & 0xFF == ord('s'):
                if x is not None and y is not None and z is not None:
                    result_array = calculate_ik(x, y, z)
                    if result_array:
                        print("Ters kinematik sonuçları:")
                        for angles in result_array:
                            send_angles_via_uart(angles)
                    else:
                        print("Geçerli bir çözüm bulunamadı.")
                else:
                    print("XYZ değerleri geçerli değil.")

            # 'q' tuşuna basıldığında çıkış yapılır
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Kamerayı durdur
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
