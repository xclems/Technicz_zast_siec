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
        self.lr = 0.001
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
        self.root.geometry("1100x850")
        self.root.minsize(950, 650)
        self.root.configure(bg="#000000")

        self.grid_size = 16
        self.nn = DeepNeuralNetwork()
        self.drawing_data = np.zeros((self.grid_size, self.grid_size))
        self.dataset = []
        self.manual_counts = [0, 0, 0]
        self.dataset_file = "dataset.pkl"

        self.test_dataset = []
        self.test_dataset_file = "test_dataset.pkl"
        self.load_test_dataset()

        self.load_dataset()
        self.setup_ui()
        self.canvas.bind("<Configure>", lambda e: self.redraw_grid())

    def setup_ui(self):
        # Konfiguracja głównej siatki
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Obszar rysowania (Canvas)
        self.canvas = tk.Canvas(
            self.root, bg="#0a0a0a", highlightthickness=1, highlightbackground="#222"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        # Prawy panel boczny
        side = tk.Frame(self.root, bg="#000000", padx=15)
        side.grid(row=0, column=1, sticky="nsew")

        # --- SEKCJA: STATYSTYKI ---
        stats_frame = tk.LabelFrame(
            side,
            text=" Statystyki ",
            fg="#888",
            bg="#000000",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=10,
        )
        stats_frame.pack(fill="x", pady=5)

        self.total_lbl = tk.Label(
            stats_frame,
            text="BAZA: 0",
            fg="#00ff41",
            bg="#000000",
            font=("Courier", 12, "bold"),
        )
        self.total_lbl.pack()

        self.class_stat_lbl = tk.Label(
            stats_frame,
            text="P-0 R-0 O-0",
            fg="white",
            bg="#000000",
            font=("Courier", 10),
            justify="left",
        )
        self.class_stat_lbl.pack(pady=5)

        # --- SEKCJA: DODAWANIE PRÓBEK ---
        tk.Label(
            side,
            text="DODAJ PRÓBKI (kliknięcia):",
            fg="#888",
            bg="#000000",
            font=("Arial", 8, "bold"),
        ).pack(anchor="w", pady=(10, 0))

        # Ramka na przyciski dodawania (Naprawiony NameError)
        add_frame = tk.Frame(side, bg="#000000")
        add_frame.pack(fill="x", pady=5)

        self.manual_labels = {}
        for i, char in enumerate(["P", "R", "O"]):
            f = tk.Frame(add_frame, bg="#111", padx=5, pady=5)
            f.pack(fill="x", pady=2)
            tk.Button(
                f,
                text=f"Dodaj {char}",
                bg="#222",
                fg="white",
                font=("Arial", 9, "bold"),
                width=12,
                command=lambda idx=i: self.add_sample(idx),
            ).pack(side="left")
            lbl = tk.Label(
                f, text="0", fg="#00ff41", bg="#111", font=("Arial", 10, "bold")
            )
            lbl.pack(side="right", padx=10)
            self.manual_labels[i] = lbl

        # --- SEKCJA: TRYB ZAPISU ---
        mode_frame = tk.LabelFrame(
            side,
            text=" TRYB ZAPISU ",
            fg="#888",
            bg="#000000",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=10,
        )
        mode_frame.pack(fill="x", pady=10)

        self.mode_var = tk.StringVar(value="train")
        tk.Radiobutton(
            mode_frame,
            text="Uczenie (Trening)",
            variable=self.mode_var,
            value="train",
            bg="#000000",
            fg="white",
            selectcolor="#222",
            activebackground="#333333",
            activeforeground="#00ff41",
        ).pack(anchor="w")
        tk.Radiobutton(
            mode_frame,
            text="Testowanie (Zestaw testowy)",
            variable=self.mode_var,
            value="test",
            bg="#000000",
            fg="white",
            selectcolor="#222",
            activebackground="#333333",
            activeforeground="#00ff41",
        ).pack(anchor="w")

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=15)

        # --- SEKCJA: TRENOWANIE ---
        tk.Label(side, text="EPOKI", fg="white", bg="#000000").pack()
        self.epoch_var = tk.IntVar(value=5)
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

        tk.Button(
            side,
            text="Statystyki zestawu testowego",
            bg="#3498db",
            fg="white",
            command=self.show_accuracy_stats,
        ).pack(fill="x", pady=5)

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=15)

        # --- GŁÓWNE PRZYCISKI AKCJI ---
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
            side, text="WYCZYŚĆ", bg="#222", fg="white", command=self.clear_canvas
        ).pack(fill="x", pady=2)

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=10)

        tk.Button(
            side,
            text="USUŃ BAZĘ",
            bg="#c0392b",
            fg="white",
            command=self.reset_all_data,
        ).pack(fill="x", pady=2)

        tk.Button(
            side,
            text="USUŃ BAZĘ TESTOWĄ",
            bg="#e67e22",
            fg="white",
            command=self.reset_test_data,
        ).pack(fill="x", pady=2)

        tk.Button(
            side,
            text="RESET WAGI",
            bg="#444",
            fg="white",
            command=self.reset_weights_action,
        ).pack(fill="x", pady=2)

        # Aktualizacja wyświetlania statystyk
        self.update_stat_display()

    def draw(self, event):
        self.canvas.config(highlightbackground="#222")

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

    def load_test_dataset(self):
        if os.path.exists(self.test_dataset_file):
            try:
                with open(self.test_dataset_file, "rb") as f:
                    self.test_dataset = pickle.load(f)
            except:
                self.test_dataset = []

    def save_test_dataset(self):
        with open(self.test_dataset_file, "wb") as f:
            pickle.dump(self.test_dataset, f)

    def add_sample(self, label):
        if self.mode_var.get() == "test":
            self.add_to_test(label)
        else:
            self.add_to_dataset(label)

    def add_to_test(self, label):
        processed = self.process_image(self.drawing_data)
        if processed is None:
            return
        self.test_dataset.append((processed.flatten(), label))
        self.save_test_dataset()
        self.clear_canvas()
        self.update_stat_display()

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
            "LOSS",
            color="white",
            fontsize=12,
            pad=15,
            fontweight="bold",
        )
        ax.set_xlabel("Epochs", color="#888", fontsize=10)
        ax.set_ylabel("Error", color="#888", fontsize=10)
        ax.tick_params(colors="white", labelsize=9)

        for spine in ax.spines.values():
            spine.set_color("#444444")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def show_accuracy_stats(self):
        if not self.test_dataset:
            messagebox.showwarning("!", "Nie ma danych testowych!")
            return

        correct = 0
        total = len(self.test_dataset)

        class_total = [0, 0, 0]
        class_correct = [0, 0, 0]
        labels = ["P", "R", "O"]

        for x, y_true in self.test_dataset:
            out = self.nn.forward(x.reshape(1, -1))
            y_pred = np.argmax(out)
            class_total[y_true] += 1
            if y_pred == y_true:
                correct += 1
                class_correct[y_true] += 1

        accuracy = (correct / total) * 100

        percents = [
            (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
            for i in range(3)
        ]

        win = tk.Toplevel(self.root)
        win.title("Statystyki Dokładności")
        win.configure(bg="#000000")

        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#000000")
        ax.set_facecolor("#000000")

        bars = ax.bar(
            labels, percents, color="#00ff41", edgecolor="white", linewidth=0.5
        )

        ax.set_ylim(0, 105)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5, axis="y")

        ax.set_title(
            f"DOKŁADNOŚĆ OGÓLNA: {accuracy:.1f}%",
            color="white",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel("Procent (%)", color="#888", fontsize=10)
        ax.tick_params(colors="white", labelsize=10)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 2,
                f"{int(height)}%",
                ha="center",
                va="bottom",
                color="white",
                fontsize=10,
            )

        for spine in ax.spines.values():
            spine.set_color("#444444")

        fig.tight_layout()

        # Виведення в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Також виведемо текстове повідомлення, як і раніше
        messagebox.showinfo(
            "Wynik testu",
            f"Ogólna dokładność: {accuracy:.1f}%\nPoprawne: {correct} z {total}",
        )

    def predict(self):
        img = self.process_image(self.drawing_data)
        if img is None:
            messagebox.showwarning("!", "Narysuj coś na płótnie")
            return

        self.nn.forward(img.flatten().reshape(1, -1), T=1.0)
        logits = self.nn.z2[0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        idx = np.argmax(probs)
        max_prob = probs[idx]

        sorted_probs = np.sort(probs)
        diff = sorted_probs[-1] - sorted_probs[-2]
        is_unsure = max_prob < 0.60 or diff < 0.25

        if is_unsure:
            self.canvas.config(highlightbackground="red")
            messagebox.showwarning(
                "Wynik",
                f"Niepewny wynik.\n"
                f"Prawdopodobieństwo: {max_prob * 100:.1f}%\n"
                f"Różnica klas: {diff * 100:.1f}%",
            )
        else:
            self.canvas.config(highlightbackground="#00ff41")
            res_txt = ["P", "R", "O"]
            messagebox.showinfo(
                "Wynik", f"Rozpoznano: {res_txt[idx]}\nPewność: {max_prob * 100:.1f}%"
            )

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

    def reset_test_data(self):
        if messagebox.askyesno("?", "Usunąć bazę testową?"):
            self.test_dataset = []
            self.update_stat_display()
            if os.path.exists(self.test_dataset_file):
                os.remove(self.test_dataset_file)
            messagebox.showinfo("!", "Baza testowa została usunięta")

    def update_stat_display(self):
        counts = [0, 0, 0]
        for _, y in self.dataset:
            counts[y] += 1

        test_counts = [0, 0, 0]
        for _, y in self.test_dataset:
            test_counts[y] += 1

        self.total_lbl.config(
            text=f"BAZA: {len(self.dataset)} | Test: {len(self.test_dataset)}"
        )
        self.class_stat_lbl.config(
            text=f"TRAIN: P-{counts[0]} R-{counts[1]} O-{counts[2]}\n"
            f"TEST:  P-{test_counts[0]} R-{test_counts[1]} O-{test_counts[2]}"
        )

        for i in range(3):
            self.manual_labels[i].config(text=str(self.manual_counts[i]))

    def clear_canvas(self):
        self.drawing_data.fill(0)
        self.redraw_grid()
        self.canvas.config(highlightbackground="#222")

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
