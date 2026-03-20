import os
import pickle
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import rotate, zoom  # Dodano zoom dla skalowania


class DeepNeuralNetwork:
    def __init__(self):
        self.input_size = 784
        self.hidden_size = 128
        self.output_size = 3
        self.lr = 0.01
        self.weights_file = "model_weights.pkl"
        self.history = []
        self.reset_weights()
        self.load_model()

    def reset_weights(self):
        # Inicjalizacja wag (He initialization)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(
            2 / self.input_size
        )
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(
            2 / self.hidden_size
        )
        if hasattr(self, "history"):
            self.history = []

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.hidden = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.hidden, self.W2)
        self.output = self.softmax(self.z2)
        return self.output

    def train_step(self, X, y_label):
        target = np.zeros((1, 3))
        target[0, y_label] = 1
        output = self.forward(X)
        error = output - target
        loss = np.mean(np.square(error))

        d_W2 = self.hidden.T.dot(error)
        error_hidden = error.dot(self.W2.T) * self.leaky_relu_derivative(self.hidden)
        d_W1 = X.T.dot(error_hidden)

        self.W2 -= self.lr * d_W2
        self.W1 -= self.lr * d_W1
        return loss

    def save_model(self):
        with open(self.weights_file, "wb") as f:
            pickle.dump((self.W1, self.W2), f)

    def load_model(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "rb") as f:
                    self.W1, self.W2 = pickle.load(f)
            except:
                pass


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PRO AI - Professional Neural Suite v4")
        self.root.geometry("1000x750")
        self.root.configure(bg="#1e1e1e")

        self.nn = DeepNeuralNetwork()
        self.drawing_data = np.zeros((28, 28))
        self.dataset = []
        self.dataset_file = "dataset.pkl"
        self.load_dataset()
        self.setup_ui()

    def setup_ui(self):
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- PANEL RYSOWANIA ---
        canvas_container = tk.Frame(self.root, bg="#1e1e1e", padx=20, pady=20)
        canvas_container.grid(row=0, column=0, sticky="nsew")
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_container,
            bg="#ffffff",
            highlightthickness=2,
            highlightbackground="#4a90e2",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<B1-Motion>", self.draw)

        btn_frame = tk.Frame(canvas_container, bg="#1e1e1e", pady=15)
        btn_frame.grid(row=1, column=0, sticky="ew")

        tk.Button(
            btn_frame,
            text="SPRAWDŹ",
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.predict,
            padx=25,
        ).pack(side="left", padx=10)
        tk.Button(
            btn_frame,
            text="WYCZYŚĆ EKRAN",
            bg="#d35400",
            fg="white",
            command=self.clear_canvas,
        ).pack(side="left", padx=10)

        # --- PANEL BOCZNY ---
        side_panel = tk.Frame(self.root, bg="#2d2d2d", padx=20, pady=20)
        side_panel.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            side_panel,
            text="1. DANE (Licznik x3)",
            fg="#4a90e2",
            bg="#2d2d2d",
            font=("Arial", 11, "bold"),
        ).pack(pady=(0, 10))
        self.stat_labels = {}
        for i, char in enumerate(["P", "R", "O"]):
            f = tk.Frame(side_panel, bg="#2d2d2d")
            f.pack(fill="x", pady=3)
            tk.Button(
                f,
                text=f"Dodaj {char}",
                bg="#444",
                fg="white",
                width=10,
                command=lambda idx=i: self.add_to_dataset(idx),
            ).pack(side="left")
            lbl = tk.Label(
                f, text="0", fg="#2ecc71", bg="#2d2d2d", font=("Arial", 10, "bold")
            )
            lbl.pack(side="right", padx=10)
            self.stat_labels[i] = lbl

        ttk.Separator(side_panel, orient="horizontal").pack(fill="x", pady=20)

        tk.Label(
            side_panel,
            text="2. TRENING",
            fg="#4a90e2",
            bg="#2d2d2d",
            font=("Arial", 11, "bold"),
        ).pack()
        self.epoch_var = tk.IntVar(value=50)
        tk.Spinbox(
            side_panel, from_=1, to=2000, textvariable=self.epoch_var, width=10
        ).pack(pady=5)
        tk.Button(
            side_panel,
            text="TRENUJ",
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.run_training_session,
        ).pack(fill="x", pady=5)

        ttk.Separator(side_panel, orient="horizontal").pack(fill="x", pady=20)

        tk.Button(
            side_panel,
            text="WYKRESY",
            bg="#9b59b6",
            fg="white",
            command=self.show_stats,
        ).pack(fill="x", pady=3)
        tk.Button(
            side_panel,
            text="USUŃ BAZĘ",
            bg="#c0392b",
            fg="white",
            command=self.reset_all_data,
        ).pack(fill="x", pady=3)
        tk.Button(
            side_panel,
            text="RESETUJ WAGI",
            bg="#7f8c8d",
            fg="white",
            command=self.reset_weights_action,
        ).pack(fill="x", pady=3)

        self.update_stat_display()

    def draw(self, event):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = int(event.x / (w / 28)), int(event.y / (h / 28))
        if 0 <= x < 28 and 0 <= y < 28:
            self.drawing_data[y, x] = 1.0
            r = 8
            self.canvas.create_oval(
                event.x - r,
                event.y - r,
                event.x + r,
                event.y + r,
                fill="black",
                outline="black",
            )

    def process_image(self, data):
        # 1. Znajdź Bounding Box
        rows = np.any(data, axis=1)
        cols = np.any(data, axis=0)
        if not np.any(rows):
            return np.zeros((28, 28))

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = data[rmin : rmax + 1, cmin : cmax + 1]

        # 2. Skalowanie (Stretch to fit) do 20x20 (zostawiamy margines)
        target_h, target_w = 20, 20
        h, w = cropped.shape
        zoom_h = target_h / h
        zoom_w = target_w / w

        # Używamy zoom do rozciągnięcia litery
        stretched = zoom(cropped, (zoom_h, zoom_w), order=1)

        # 3. Umieszczenie w ramce 28x28 (centrowanie)
        final_img = np.zeros((28, 28))
        start_h = (28 - target_h) // 2
        start_w = (28 - target_w) // 2
        final_img[start_h : start_h + target_h, start_w : start_w + target_w] = (
            stretched
        )

        return np.where(final_img > 0.1, 1.0, 0.0)  # Binaryzacja po skalowaniu

    def add_to_dataset(self, label):
        img = self.process_image(self.drawing_data)
        # Dodajemy oryginał i obroty (Augmentacja)
        self.dataset.append((img.flatten(), label))
        for angle in [-15, 15]:  # Zwiększyłem kąt dla lepszej odporności
            rot_img = rotate(img, angle, reshape=False, order=1)
            self.dataset.append((np.where(rot_img > 0.2, 1.0, 0.0).flatten(), label))

        self.save_dataset()
        self.update_stat_display()
        self.clear_canvas()

    def run_training_session(self):
        if len(self.dataset) < 3:
            return
        epochs = self.epoch_var.get()
        self.nn.history = []
        for _ in range(epochs):
            loss = 0
            np.random.shuffle(self.dataset)
            for x, y in self.dataset:
                loss += self.nn.train_step(x.reshape(1, -1), y)
            self.nn.history.append(loss / len(self.dataset))
        self.nn.save_model()
        messagebox.showinfo("OK", "Trening zakończony!")

    def predict(self):
        img = self.process_image(self.drawing_data)
        out = self.nn.forward(img.flatten().reshape(1, -1))
        idx = np.argmax(out)
        chars = ["P", "R", "O"]
        messagebox.showinfo(
            "Wynik", f"To jest: {chars[idx]} ({out[0][idx] * 100:.1f}%)"
        )

    def reset_weights_action(self):
        if messagebox.askyesno("Reset", "Czy na pewno wyczyścić wagi (pamięć sieci)?"):
            self.nn.reset_weights()
            if os.path.exists(self.nn.weights_file):
                os.remove(self.nn.weights_file)
            messagebox.showinfo("Reset", "Wagi zostały zresetowane.")

    def show_stats(self):
        if not self.nn.history:
            return
        win = tk.Toplevel(self.root)
        win.title("Statystyki")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(self.nn.history)
        ax.set_title("Loss over epochs")
        ax.grid(True)
        FigureCanvasTkAgg(fig, master=win).get_tk_widget().pack()

    def update_stat_display(self):
        counts = [0, 0, 0]
        for _, y in self.dataset:
            counts[y] += 1
        for i in range(3):
            self.stat_labels[i].config(text=str(counts[i]))

    def reset_all_data(self):
        if messagebox.askyesno("Uwaga", "Usunąć bazę?"):
            self.dataset = []
            if os.path.exists(self.dataset_file):
                os.remove(self.dataset_file)
            self.update_stat_display()

    def save_dataset(self):
        with open(self.dataset_file, "wb") as f:
            pickle.dump(self.dataset, f)

    def load_dataset(self):
        if os.path.exists(self.dataset_file):
            with open(self.dataset_file, "rb") as f:
                self.dataset = pickle.load(f)

    def clear_canvas(self):
        self.drawing_data.fill(0)
        self.canvas.delete("all")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
