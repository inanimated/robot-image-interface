import tkinter as tk
from logging import root
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSlider, QLineEdit, QLabel
from PyQt5.uic import loadUi

x,y,z = None, None, None

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
    global current_thetas
    try:
        result = inverse_kinematics_xyz(x, y, z)

    #  if not result:
    #     messagebox.showinfo("Sonuç", "Geçerli bir çözüm bulunamadı.")
    #    else:
    #       result_text = "\n".join(
    #            [", ".join([f"{rad_to_deg(angle):.2f}°" for angle in angles]) for angles in result]
    #       )
    #        messagebox.showinfo("Sonuç", f"Geçerli Eklem Açıları:\n{result_text}")

        if result:
            first_solution = result[0]
            result_text = ", ".join([f"{rad_to_deg(angle):.2f}" for angle in first_solution])
            first_element = result_text.split(", ")[0]
            print("Birinci Açı:", first_element)
            second_element = result_text.split(", ")[1]
            print("İkinci Açı:", second_element)
            third_element = result_text.split(", ")[2]
            print("Üçüncü Açı:", third_element)
            fourth_element = result_text.split(", ")[3]
            print("Dördüncü Açı:", fourth_element)
            fifth_element = result_text.split(", ")[4]
            print("Beşinci Açı:", fifth_element)
            sixth_element = result_text.split(", ")[5]
            print("Altıncı Açı:", sixth_element)
            print("Sonuç", f"İlk Geçerli Eklem Açıları:\n{result_text}")
            #messagebox.showinfo("Sonuç", f"İlk Geçerli Eklem Açıları:\n{result_text}")
            return result_text

        else:
            print("Sonuç", "Geçerli bir çözüm bulunamadı.")
            #messagebox.showinfo("Sonuç", "Geçerli bir çözüm bulunamadı.")


            # İlk çözümü alıp, grafiği güncelle
            plot_robot_3d(result[0])
            current_thetas = result[0]
    except Exception as e:
        print("Hata", f"Girişlerde bir hata var: {e}")
        #messagebox.showerror("Hata", f"Girişlerde bir hata var: {e}")





# DH parametreleri: alpha, a, d, theta
DH_parameters = [
    [0, 0, 0, 0],
    [-90, 0, 0, -90],
    [0, 246, 0, 0],
    [-90, 28, 200, 0],
    [90, 0, 0, 0],
    [-90, 0, 100, 0]
]

# İleri Kinematik hesaplama fonksiyonu
def forward_kinematics(thetas):
    T_k = np.eye(4)
    points = [T_k[:3, 3]]
    for i in range(len(DH_parameters)):
        alpha, a, d, theta_offset = DH_parameters[i]
        theta = np.degrees(thetas[i]) + theta_offset
        T_current = np.array([
            [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0, a],
            [np.sin(np.radians(theta)) * np.cos(np.radians(alpha)),
             np.cos(np.radians(theta)) * np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)),
             -np.sin(np.radians(alpha)) * d],
            [np.sin(np.radians(theta)) * np.sin(np.radians(alpha)),
             np.cos(np.radians(theta)) * np.sin(np.radians(alpha)), np.cos(np.radians(alpha)),
             np.cos(np.radians(alpha)) * d],
            [0, 0, 0, 1]
        ])
        T_k = T_k @ T_current
        points.append(T_k[:3, 3])
    return np.array(points)

# 3D robot grafiğini çizdiren fonksiyon
def plot_robot_3d(thetas):
    points = forward_kinematics(thetas)
    ax.cla()
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'bo-', linewidth=3)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_zlim(-100, 500)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Robot")

    fig.canvas.draw()


    # Yörünge planlama fonksiyonu
def trajectory_planning(start_thetas, end_thetas, steps=50):
    global current_thetas
    theta_trajectory = np.linspace(start_thetas, end_thetas, steps)
    for theta in theta_trajectory:
        plot_robot_3d(theta)
        time.sleep(0.1)
        root.update()

# Yörünge Planla butonuna basıldığında çalışacak fonksiyon
def plan_trajectory():
    global current_thetas
    try:
        x = int(entry_x.get())
        y = int(entry_y.get())
        z = int(entry_z.get())

        result = inverse_kinematics_xyz(x, y, z)

        if not result:
            print("Sonuç", "Geçerli bir çözüm bulunamadı.")
        else:
            end_thetas = result[0]
            trajectory_planning(current_thetas, end_thetas)  # Mevcut konumdan hedef konuma yörünge planla
            current_thetas = end_thetas  # Mevcut açılar güncellenir
    except Exception as e:
        print("Hata", f"Girişlerde bir hata var: {e}")

'''
#Tkinter arayüzünü oluşturma

root = tk.Tk()
root.title("Ters Kinematik Hesaplama ve Yörünge Planlama")

# Sol panel
frame_left = ttk.Frame(root, padding="10 10 10 10")
frame_left.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame_left, text="X:").grid(column=0, row=0, sticky=tk.W)
entry_x = ttk.Entry(frame_left, width=7)
entry_x.grid(column=1, row=0, sticky=(tk.W, tk.E))

ttk.Label(frame_left, text="Y:").grid(column=0, row=1, sticky=tk.W)
entry_y = ttk.Entry(frame_left, width=7)
entry_y.grid(column=1, row=1, sticky=(tk.W, tk.E))

ttk.Label(frame_left, text="Z:").grid(column=0, row=2, sticky=tk.W)
entry_z = ttk.Entry(frame_left, width=7)
entry_z.grid(column=1, row=2, sticky=(tk.W, tk.E))

calculate_button = ttk.Button(frame_left, text="Hesapla", command=calculate_ik)
calculate_button.grid(column=0, row=3, columnspan=2, pady=10)

trajectory_button = ttk.Button(frame_left, text="Yörünge Planla", command=plan_trajectory)
trajectory_button.grid(column=0, row=4, columnspan=2, pady=10)

# Üst panel
frame_top = ttk.Frame(root, padding="10 10 10 10")
frame_top.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

# Matplotlib figürü
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D grafiği Tkinter ile bütünleştirme
canvas = FigureCanvasTkAgg(fig, master=frame_top)
canvas.get_tk_widget().grid(row=0, column=0)

# Başlangıçtaki robot konumu
current_thetas = np.zeros(6)
plot_robot_3d(current_thetas, )

root.mainloop()'''



