import tkinter as tk
from tkinter import ttk
from tkinter import colorchooser
import numpy as np
import torch
from envs.collage_env_hybrid import CollageEnv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

IMAGE_SIZE = 128
NUM_SHAPES = 20
ACTION_DIM = 13  # 根据你的动作空间实际维度调整

target_image_path = "assets/targets/target2.png"  # 替换为你的目标图像路径

env = CollageEnv(
    target_images=[target_image_path],
    num_shapes=NUM_SHAPES,
    canvas_size=IMAGE_SIZE,
    training=True
)

num_logits = 4  # 或你的实际 logits 数
ACTION_NAMES = [f"shape_logits[{i}]" for i in range(num_logits)] + ["tx", "ty", "sx", "sy", "angle_sin", "angle_cos", "r", "g", "b"]
ACTION_NAMES = ACTION_NAMES[:ACTION_DIM]



class ActionEditorGUI:
    def __init__(self, master):
        self.master = master
        master.title("动作编辑与奖励验证")
        self.env = env  # 提前赋值，放在最前面
        self.action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.step_count = 0

        self.scales = []
        self.labels = []

        # 先创建除 angle_sin/angle_cos 外的滑块
        for i, name in enumerate(ACTION_NAMES):
            if name in ("angle_sin", "angle_cos"):
                self.scales.append(None)
                self.labels.append(None)
                continue
            frame = ttk.Frame(master)
            frame.pack()
            label = ttk.Label(frame, text=f"{name}: {self.action[i]:.2f}")
            label.pack(side=tk.LEFT)
            scale = ttk.Scale(frame, from_=-1.0, to=1.0, orient=tk.HORIZONTAL, length=200,
                              command=lambda val, idx=i: self.on_slider_change(idx, val))
            scale.set(0.0)
            scale.pack(side=tk.LEFT)
            self.scales.append(scale)
            self.labels.append(label)

        # 单独添加角度滑块
        angle_frame = ttk.Frame(master)
        angle_frame.pack()
        ttk.Label(angle_frame, text="角度(°)").pack(side=tk.LEFT)
        self.angle_var = tk.DoubleVar(value=0)
        self.angle_scale = ttk.Scale(angle_frame, from_=-180, to=180, orient=tk.HORIZONTAL, length=200,
                                     variable=self.angle_var, command=lambda v: self.set_angle(float(v)))
        self.angle_scale.pack(side=tk.LEFT)

        # 目标图像透明度滑块
        alpha_frame = ttk.Frame(master)
        alpha_frame.pack()
        ttk.Label(alpha_frame, text="目标图像透明度").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.alpha_scale = ttk.Scale(alpha_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=200,
                                     variable=self.alpha_var, command=lambda v: self.show_canvas())
        self.alpha_scale.pack(side=tk.LEFT)

        # 按钮
        self.apply_button = ttk.Button(master, text="应用动作", command=self.apply_action)
        self.apply_button.pack()
        self.reset_button = ttk.Button(master, text="重置环境", command=self.reset_env)
        self.reset_button.pack()
        self.save_button = ttk.Button(master, text="保存轨迹", command=self.save_trajectory)
        self.save_button.pack()

        # 显示reward
        self.reward_var = tk.StringVar()
        self.reward_label = ttk.Label(master, textvariable=self.reward_var)
        self.reward_label.pack()

        # matplotlib画布
        self.fig, self.ax = plt.subplots(figsize=(6,6))  # 或 (8,8)
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master)
        self.canvas_widget.get_tk_widget().pack()

        # 预存目标图像
        self.reset_env()
        arr = self.env.target.detach().cpu().clamp(0, 1).numpy()
        if arr.ndim == 3:
            self.target_img = arr.transpose(1, 2, 0)
        elif arr.ndim == 2:
            self.target_img = arr
        else:
            raise ValueError(f"Unexpected target shape: {arr.shape}")


        self.trajectory = []  # 用于保存轨迹
        self.last_obs, _ = self.env.reset()  # 保存上一次的obs

        # 取色器按钮
        self.color_button = ttk.Button(master, text="取色器", command=self.choose_color)
        self.color_button.pack()

        # 16进制颜色输入
        hex_frame = ttk.Frame(master)
        hex_frame.pack()
        ttk.Label(hex_frame, text="16进制颜色:").pack(side=tk.LEFT)
        self.hex_color_var = tk.StringVar()
        self.hex_entry = ttk.Entry(hex_frame, textvariable=self.hex_color_var, width=10)
        self.hex_entry.pack(side=tk.LEFT)
        self.hex_entry.bind("<Return>", self.on_hex_color_enter)

    def update_action(self, idx, val):
        if idx >= len(self.action):
            return
        self.action[idx] = float(val)
        if idx < len(self.labels) and self.labels[idx] is not None:
            self.labels[idx].config(text=f"{ACTION_NAMES[idx]}: {float(val):.2f}")

    def on_slider_change(self, idx, val):
        # 跳过 angle_sin/angle_cos 的滑块（已合并为角度滑块）
        if ACTION_NAMES[idx] in ("angle_sin", "angle_cos"):
            return
        self.update_action(idx, val)
        self.preview_action()

    def preview_action(self):
        canvas_backup = self.env.canvas.clone()
        obs, reward, terminated, truncated, _ = self.env.step(self.action)
        self.show_canvas()
        self.env.canvas = canvas_backup

    def apply_action(self):
        obs = self.last_obs
        action = self.action.copy()
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        # 保存轨迹
        self.trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done
        })
        self.reward_var.set(f"Reward: {reward:.6f}")
        self.show_canvas()
        self.step_count += 1
        self.last_obs = next_obs

    def reset_env(self):
        obs, _ = self.env.reset()
        self.last_obs = obs
        self.reward_var.set("Reward: N/A")
        arr = self.env.target.detach().cpu().clamp(0, 1).numpy()
        if arr.ndim == 3:
            self.target_img = arr.transpose(1, 2, 0)
        elif arr.ndim == 2:
            self.target_img = arr
        else:
            raise ValueError(f"Unexpected target shape: {arr.shape}")
        self.show_canvas()
        self.step_count = 0

    def show_canvas(self):
        canvas = self.env.canvas.detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
        alpha = self.alpha_var.get()
        # 混合目标图像和当前画布
        blended = (1 - alpha) * canvas + alpha * self.target_img
        self.ax.clear()
        self.ax.imshow(blended)
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas_widget.draw()

    def choose_color(self):
        color = colorchooser.askcolor(title="选择颜色")
        if color[0] is not None:
            r, g, b = [v / 255.0 for v in color[0]]
            r_idx = ACTION_NAMES.index("r")
            g_idx = ACTION_NAMES.index("g")
            b_idx = ACTION_NAMES.index("b")
            self.action[r_idx] = r * 2 - 1  # 如果动作空间是-1~1
            self.action[g_idx] = g * 2 - 1
            self.action[b_idx] = b * 2 - 1
            self.scales[r_idx].set(self.action[r_idx])
            self.scales[g_idx].set(self.action[g_idx])
            self.scales[b_idx].set(self.action[b_idx])
            self.labels[r_idx].config(text=f"{ACTION_NAMES[r_idx]}: {self.action[r_idx]:.2f}")
            self.labels[g_idx].config(text=f"{ACTION_NAMES[g_idx]}: {self.action[g_idx]:.2f}")
            self.labels[b_idx].config(text=f"{ACTION_NAMES[b_idx]}: {self.action[b_idx]:.2f}")
            self.preview_action()

    def on_hex_color_enter(self, event=None):
        hex_color = self.hex_color_var.get().strip()
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                r_idx = ACTION_NAMES.index("r")
                g_idx = ACTION_NAMES.index("g")
                b_idx = ACTION_NAMES.index("b")
                self.action[r_idx] = r * 2 - 1
                self.action[g_idx] = g * 2 - 1
                self.action[b_idx] = b * 2 - 1
                self.scales[r_idx].set(self.action[r_idx])
                self.scales[g_idx].set(self.action[g_idx])
                self.scales[b_idx].set(self.action[b_idx])
                self.labels[r_idx].config(text=f"{ACTION_NAMES[r_idx]}: {self.action[r_idx]:.2f}")
                self.labels[g_idx].config(text=f"{ACTION_NAMES[g_idx]}: {self.action[g_idx]:.2f}")
                self.labels[b_idx].config(text=f"{ACTION_NAMES[b_idx]}: {self.action[b_idx]:.2f}")
                self.preview_action()
            except Exception as e:
                print("颜色格式错误:", e)
        else:
            print("请输入6位16进制颜色，如 #FF00FF")

    #def save_trajectory(self):
    #    import pickle
    #    with open("expert_trajectory.pkl", "wb") as f:
    #        pickle.dump(self.trajectory, f)
    #    print("轨迹已保存到 expert_trajectory.pkl")

    def save_trajectory(self):
        obs = np.array([item["obs"] for item in self.trajectory])
        actions = np.array([item["action"] for item in self.trajectory])
        rewards = np.array([item["reward"] for item in self.trajectory])
        next_obs = np.array([item["next_obs"] for item in self.trajectory])
        dones = np.array([item["done"] for item in self.trajectory])
        episode_starts = np.zeros_like(dones)  # 如果需要episode起始标记，可自定义
        np.savez("expert_trajectory_manual.npz", obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones, episode_starts=episode_starts)
        print("轨迹已保存到 expert_trajectory_manual.npz")

    def set_angle(self, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        sin_idx = ACTION_NAMES.index("angle_sin")
        cos_idx = ACTION_NAMES.index("angle_cos")
        self.action[sin_idx] = np.sin(angle_rad) 
        self.action[cos_idx] = np.cos(angle_rad)
        # 显示当前值
        if self.labels[sin_idx] is not None:
            self.labels[sin_idx].config(text=f"angle_sin: {self.action[sin_idx]:.2f}")
        if self.labels[cos_idx] is not None:
            self.labels[cos_idx].config(text=f"angle_cos: {self.action[cos_idx]:.2f}")
        self.preview_action()

if __name__ == "__main__":
    root = tk.Tk()
    gui = ActionEditorGUI(root)
    root.mainloop()