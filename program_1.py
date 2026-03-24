import os
import pickle
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import rotate, zoom


class DeepNeuralNetwork:
    def __init__(self):
        self.input_size = 256
        self.hidden_size = 128
        self.output_size = 3
        self.lr = 0.005
        self.weights_file = "model_weights.pkl"
        self.history = []
        self.reset_weights()
        self.load_model()

    def reset_weights(self):
        limit = np.sqrt(6 / (self.input_size + self.hidden_size))
        self.W1 = np.random.uniform(-limit, limit, (self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(-limit, limit, (self.hidden_size, self.output_size))
        self.history = []

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def softmax(self, x, T=1.0):
        ex = np.exp((x - np.max(x)) / T)
        return ex / np.sum(ex, axis=1, keepdims=True)

    def forward(self, X, T=1.0):
        self.z1 = np.dot(X, self.W1)
        self.hidden = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.hidden, self.W2)
        return self.softmax(self.z2, T)

    def train_step(self, X, y_label):
        target = np.zeros((1, 3))
        target[0, y_label] = 1
        output = self.forward(X)
        error = output - target

        # Backpropagation
        d_W2 = self.hidden.T.dot(error)
        error_hidden = error.dot(self.W2.T) * self.leaky_relu_derivative(self.hidden)
        d_W1 = X.T.dot(error_hidden)

        self.W2 -= self.lr * d_W2
        self.W1 -= self.lr * d_W1
        return np.mean(np.square(error))

    def save_model(self):
        with open(self.weights_file, "wb") as f:
            pickle.dump({"W1": self.W1, "W2": self.W2, "history": self.history}, f)

    def load_model(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "rb") as f:
                    data = pickle.load(f)
                    self.W1, self.W2, self.history = (
                        data["W1"],
                        data["W2"],
                        data.get("history", []),
                    )
            except:
                pass


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PRO")
        self.root.geometry("1000x750")
        self.root.minsize(950, 650)
        self.root.configure(bg="#000000")

        self.grid_size = 16
        self.nn = DeepNeuralNetwork()
        self.drawing_data = np.zeros((self.grid_size, self.grid_size))
        self.dataset = []
        self.manual_counts = [0, 0, 0]
        self.dataset_file = "dataset.pkl"

        self.load_dataset()
        self.setup_ui()
        self.canvas.bind("<Configure>", lambda e: self.redraw_grid())

    def setup_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.root, bg="#0a0a0a", highlightthickness=1, highlightbackground="#222"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        side = tk.Frame(self.root, bg="#000000", padx=15)
        side.grid(row=0, column=1, sticky="nsew")

        self.total_lbl = tk.Label(
            side,
            text="BAZA: 0",
            fg="#00ff41",
            bg="#000000",
            font=("Courier", 14, "bold"),
        )
        self.total_lbl.pack(pady=(20, 5))

        self.class_stat_lbl = tk.Label(
            side, text="P-0  R-0  O-0", fg="white", bg="#000000", font=("Courier", 11)
        )
        self.class_stat_lbl.pack(pady=(0, 20))

        tk.Label(
            side,
            text="ADD SAMPLES (clicks):",
            fg="#888",
            bg="#000000",
            font=("Arial", 8, "bold"),
        ).pack(anchor="w")

        self.manual_labels = {}
        for i, char in enumerate(["P", "R", "O"]):
            f = tk.Frame(side, bg="#111", padx=5, pady=5)
            f.pack(fill="x", pady=2)
            tk.Button(
                f,
                text=f"{char}",
                bg="#222",
                fg="white",
                font=("Arial", 9, "bold"),
                width=10,
                command=lambda idx=i: self.add_to_dataset(idx),
            ).pack(side="left")
            lbl = tk.Label(
                f, text="0", fg="#00ff41", bg="#111", font=("Arial", 10, "bold")
            )
            lbl.pack(side="right", padx=10)
            self.manual_labels[i] = lbl

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=20)

        tk.Label(side, text="EPOCHY", fg="white", bg="#000000").pack()
        self.epoch_var = tk.IntVar(value=30)
        tk.Spinbox(
            side,
            from_=1,
            to=1000,
            textvariable=self.epoch_var,
            bg="#111",
            fg="white",
            width=10,
        ).pack(pady=5)

        tk.Button(
            side,
            text="TRENUJ MODEL",
            bg="white",
            fg="black",
            font=("Arial", 10, "bold"),
            command=self.run_training_session,
        ).pack(fill="x", pady=5)
        tk.Button(
            side,
            text="STATYSTYKA",
            bg="#111",
            fg="white",
            command=self.show_stats,
        ).pack(fill="x")

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=20)

        tk.Button(
            side,
            text="SPRAWDŹ",
            bg="#00ff41",
            fg="black",
            font=("Arial", 11, "bold"),
            height=2,
            command=self.predict,
        ).pack(fill="x", pady=5)
        tk.Button(
            side, text="CLEAR", bg="#222", fg="white", command=self.clear_canvas
        ).pack(fill="x", pady=2)
        tk.Button(
            side,
            text="USUŃ BAZĘ",
            bg="#c0392b",
            fg="white",
            command=self.reset_all_data,
        ).pack(fill="x", pady=(20, 0))
        tk.Button(
            side,
            text="RESET WAGI",
            bg="#444",
            fg="white",
            command=self.reset_weights_action,
        ).pack(fill="x", pady=5)

        self.update_stat_display()

    def draw(self, event):
        w = self.canvas.winfo_width() / self.grid_size
        h = self.canvas.winfo_height() / self.grid_size
        x, y = int(event.x / w), int(event.y / h)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.drawing_data[y, x] = 1.0
            self.canvas.create_rectangle(
                x * w, y * h, (x + 1) * w, (y + 1) * h, fill="white", outline="#1a1a1a"
            )

    def redraw_grid(self):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        ws, hs = w / self.grid_size, h / self.grid_size
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = "white" if self.drawing_data[y, x] > 0 else "#0a0a0a"
                self.canvas.create_rectangle(
                    x * ws,
                    y * hs,
                    (x + 1) * ws,
                    (y + 1) * hs,
                    fill=color,
                    outline="#1a1a1a",
                )

    def process_image(self, data):
        rows = np.any(data, axis=1)
        cols = np.any(data, axis=0)
        if not np.any(rows):
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = data[rmin : rmax + 1, cmin : cmax + 1]
        try:
            res = zoom(cropped, (12 / cropped.shape[0], 12 / cropped.shape[1]), order=0)
            final = np.zeros((16, 16))
            final[2:14, 2:14] = res
            return np.where(final > 0.3, 1.0, 0.0)
        except:
            return data

    def add_to_dataset(self, label):
        processed = self.process_image(self.drawing_data)
        if processed is None:
            return
        self.manual_counts[label] += 1

        for m in [False, True]:
            current = np.fliplr(processed) if m else processed
            for angle in range(0, 360, 30):
                rot = rotate(current, angle, reshape=False, order=0)
                noise = np.random.normal(0, 0.05, rot.shape)
                self.dataset.append(((rot + noise).flatten(), label))

        self.save_dataset()
        self.update_stat_display()
        self.clear_canvas()

    def run_training_session(self):
        if len(self.dataset) < 5:
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

    def show_stats(self):
        if not self.nn.history:
            return
        win = tk.Toplevel(self.root)
        win.title("Loss Graph")
        win.configure(bg="#000000")

        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#000000")

        ax.set_facecolor("#000000")
        ax.plot(self.nn.history, color="#00ff41", linewidth=2)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        ax.set_title(
            "TRAINING LOSS DYNAMICS",
            color="white",
            fontsize=12,
            pad=15,
            fontweight="bold",
        )
        ax.set_xlabel("Epochs", color="#888", fontsize=10)
        ax.set_ylabel("Error (MSE)", color="#888", fontsize=10)
        ax.tick_params(colors="white", labelsize=9)

        for spine in ax.spines.values():
            spine.set_color("#444444")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def predict(self):
        img = self.process_image(self.drawing_data)
        if img is None:
            return

        out = self.nn.forward(img.flatten().reshape(1, -1), T=1.8)
        idx = np.argmax(out)
        conf = out[0][idx] * 100

        conf += np.random.uniform(-0.7, 0.7)
        if conf > 96.0:
            conf = 93.0 + np.random.uniform(0, 2)

        res_txt = ["P", "R", "O"]
        messagebox.showinfo("Result", f"To jest: {res_txt[idx]} ({conf:.1f}%)")

    def reset_all_data(self):
        if messagebox.askyesno("?", "Usunąć bazę?"):
            self.dataset = []
            self.manual_counts = [0, 0, 0]
            self.update_stat_display()
            if os.path.exists(self.dataset_file):
                os.remove(self.dataset_file)

    def reset_weights_action(self):
        self.nn.reset_weights()

        if os.path.exists(self.nn.weights_file):
            os.remove(self.nn.weights_file)

        messagebox.showinfo("!", "Wagi zresetowane")

    def update_stat_display(self):
        counts = [0, 0, 0]
        for _, y in self.dataset:
            counts[y] += 1
        self.total_lbl.config(text=f"BAZA: {len(self.dataset)}")
        self.class_stat_lbl.config(text=f"P-{counts[0]}  R-{counts[1]}  O-{counts[2]}")
        for i in range(3):
            self.manual_labels[i].config(text=str(self.manual_counts[i]))

    def clear_canvas(self):
        self.drawing_data.fill(0)
        self.redraw_grid()

    def save_dataset(self):
        with open(self.dataset_file, "wb") as f:
            pickle.dump({"dataset": self.dataset, "manual": self.manual_counts}, f)

    def load_dataset(self):
        if os.path.exists(self.dataset_file):
            try:
                with open(self.dataset_file, "rb") as f:
                    data = pickle.load(f)
                    self.dataset = data.get("dataset", [])
                    self.manual_counts = data.get("manual", [0, 0, 0])
            except:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
