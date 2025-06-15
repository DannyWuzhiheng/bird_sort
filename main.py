import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Tk, filedialog, Label, Button, Toplevel, Frame
from PIL import Image, ImageTk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import threading

DATA_DIR = os.path.dirname(os.path.realpath(__file__))

class ProgressWindow:
    def __init__(self, root, main_app):
        self.main_app = main_app
        self.progress_window = Toplevel(root)
        self.progress_window.title("初始化进度")
        self.progress_window.geometry("400x225")
        self.progress_window.iconbitmap(r'./a.ico')
        self.progress_window.grab_set()
        
        self.progress = ttk.Progressbar(
            self.progress_window, 
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.progress.pack(pady=10)
        

        self.status_label = Label(self.progress_window, text="准备中...")
        self.status_label.pack(pady=5)
        

        threading.Thread(target=self.run_tasks, daemon=True).start()

    def run_tasks(self):
        tasks = [
            ("安装scikit-learn", "pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple"),
            ("安装opencv-python", "pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple"),
            ("加载数据", self.load_data),
            ("训练模型", self.train_model)
        ]

        total_steps = len(tasks)
        for step, (desc, action) in enumerate(tasks):
            self.update_status(step/total_steps*100, desc)
            if callable(action):
                action()
            else:
                os.system(f"{action} --quiet")

        self.progress_window.after(0, self.finish_initialization)

    def load_data(self):
        self.X, self.Y = [], []
        labels = os.listdir(DATA_DIR)
        total_files = sum([len(files) for r, d, files in os.walk(DATA_DIR)])
        
        processed = 0
        for label in labels:
            label_dir = os.path.join(DATA_DIR, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (256, 256))
                        hist = cv2.calcHist([img], [0,1], None, [256,256], [0,256,0,256])
                        self.X.append(hist.ravel() / hist.sum())
                        self.Y.append(label)
                    processed += 1
                    self.update_status(50 + (processed/total_files)*40, f"处理数据: {processed}/{total_files}")
                    
        self.X, self.Y = np.array(self.X), np.array(self.Y)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=42
        )
        self.knn = KNeighborsClassifier(n_neighbors=10)
        self.knn.fit(X_train, y_train)
        y_pred = self.knn.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["鸡形目", "雀形目"]))

    def update_status(self, value, text):
        self.progress_window.after(0, lambda: self._update_ui(value, text))

    def _update_ui(self, value, text):
        self.progress["value"] = value
        self.status_label.config(text=text)
        self.progress_window.update_idletasks()

    def finish_initialization(self):
        self.progress_window.destroy()
        self.main_app.show_main_window(self.knn)

class MainApp:
    def __init__(self):
        self.root = Tk()
        self.root.withdraw()
        self.progress_window = ProgressWindow(self.root, self)
        self.root.mainloop()

    def show_main_window(self, model):
        self.root.deiconify()
        ImageClassifierApp(self.root, model)

class ImageClassifierApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("鸟类分类器")
        self.root.geometry("540x720")
        self.root.iconbitmap(r'./a.ico')
        self.model = model
        self.display_image = None
        self.current_image = None

        self.setup_background()

        self.create_widgets()

    def setup_background(self):
        try:
            bg_image = Image.open("bk.png")
            bg_photo = ImageTk.PhotoImage(bg_image)
            bg_label = Label(self.root, image=bg_photo)
            bg_label.image = bg_photo  # 保持引用
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"背景加载失败: {e}")
            self.root.config(bg="#F0F0F0")

    def create_widgets(self):
        main_frame = Frame(self.root, bg="white", bd=2, relief="groove")
        main_frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=600)

        Label(main_frame, 
             text="鸟类形态分类器\n鸡形目 vs 雀形目",
             font=("微软雅黑", 16, "bold"),
             bg="white").pack(pady=15)

        self.image_panel = Label(main_frame, bg="lightgray", bd=1, relief="sunken")
        self.image_panel.pack(pady=10, fill="both", expand=True, padx=20)
        Button(main_frame, 
              text="选择图片", 
              command=self.classify_image,
              font=("宋体", 12),
              width=15).pack(pady=10)

        self.result_label = Label(main_frame, 
                                 text="等待图片选择...",
                                 font=("宋体", 14),
                                 bg="white")
        self.result_label.pack(pady=10)

    def classify_image(self):
        file_types = [
            ("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择鸟类图片",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # 读取并处理图片
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("无法读取图片文件")
                
                # 保存原始图片用于显示
                self.current_image = img.copy()
                
                # 显示处理
                self.show_image(img)
                
                # 分类处理
                self.process_classification(img)
                
            except Exception as e:
                self.result_label.config(text=f"错误: {str(e)}", fg="red")

    def show_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        max_size = (400, 400)
        pil_img.thumbnail(max_size, Image.LANCZOS)
        
        self.display_image = ImageTk.PhotoImage(pil_img)
        
        self.image_panel.config(image=self.display_image)
        self.image_panel.image = self.display_image 

    def process_classification(self, cv_img):

        processed_img = cv2.resize(cv_img, (256, 256))
        
        hist = cv2.calcHist([processed_img], [0,1], None, [256,256], [0,256,0,256])
        features = hist.ravel() / hist.sum()
        
        prediction = self.model.predict([features])[0]
        result = "鸡形目" if prediction == "2" else "雀形目"
        
        self.result_label.config(
            text=f"分类结果: {result}",
            fg="#2E7D32"
        )

if __name__ == "__main__":
    MainApp()