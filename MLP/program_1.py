import os
import pickle
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import rotate, zoom


class DeepNeuralNetwork:
    def __init__(self):
        self.input_size = 256
        self.h1_size = 12
        self.h2_size = 8
        self.output_size = 3
        self.lr = 0.01
        self.weights_file = "model_weights.pkl"
        self.history = []
        self.reset_weights()
        self.load_model()

    def reset_weights(self):
        self.W1 = np.random.randn(self.input_size, self.h1_size) * np.sqrt(
            1 / self.input_size
        )
        self.W2 = np.random.randn(self.h1_size, self.h2_size) * np.sqrt(
            1 / self.h1_size
        )
        self.W3 = np.random.randn(self.h2_size, self.output_size) * np.sqrt(
            1 / self.h2_size
        )
        self.history = []

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def softmax(self, x, T=1.0):
        ex = np.exp((x - np.max(x)) / T)
        return ex / np.sum(ex, axis=1, keepdims=True)

    def forward(self, X, T=1.0):
        self.a0 = X
        self.z1 = np.dot(self.a0, self.W1)
        self.a1 = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.a2 = self.leaky_relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3)
        return self.softmax(self.z3, T)

    def train_step(self, X, y_label):
        target = np.zeros((1, 3))
        target[0, y_label] = 1
        output = self.forward(X)
        error_out = output - target
        d_W3 = self.a2.T.dot(error_out)
        error_h2 = error_out.dot(self.W3.T) * self.leaky_relu_derivative(self.a2)
        d_W2 = self.a1.T.dot(error_h2)
        error_h1 = error_h2.dot(self.W2.T) * self.leaky_relu_derivative(self.a1)
        d_W1 = self.a0.T.dot(error_h1)

        self.W3 -= self.lr * d_W3
        self.W2 -= self.lr * d_W2
        self.W1 -= self.lr * d_W1

        return np.mean(np.square(error_out))

    def save_model(self):
        with open(self.weights_file, "wb") as f:
            pickle.dump(
                {"W1": self.W1, "W2": self.W2, "W3": self.W3, "history": self.history},
                f,
            )

    def load_model(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "rb") as f:
                    data = pickle.load(f)
                    self.W1 = data["W1"]
                    self.W2 = data["W2"]
                    self.W3 = data["W3"]
                    self.history = data.get("history", [])
            except Exception as e:
                print(f"Błąd ladowania: {e}")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PRO")
        self.root.geometry("1100x870")
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

        # --- PROGRES ---
        self.progress = ttk.Progressbar(
            side, orient="horizontal", length=200, mode="determinate"
        )
        self.progress.pack(fill="x", pady=10)

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "TProgressbar", thickness=10, background="#00ff41", troughcolor="#111"
        )

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

        self.dataset.append((processed.flatten(), label))

        for angle in [-12, -6, 6, 12]:
            rot = rotate(processed, angle, reshape=False, order=1)
            self.dataset.append((np.where(rot > 0.3, 1.0, 0.0).flatten(), label))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            shifted = np.roll(processed, shift=dx, axis=1)
            shifted = np.roll(shifted, shift=dy, axis=0)
            self.dataset.append((shifted.flatten(), label))

        noise = np.random.normal(0, 0.05, processed.shape)
        noised = np.clip(processed + noise, 0, 1)
        self.dataset.append((noised.flatten(), label))

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
            messagebox.showwarning("!", "Zamało danych dla uczenia")
            return

        def training_process():
            epochs = self.epoch_var.get()
            self.nn.history = []
            self.progress["maximum"] = epochs

            for i in range(epochs):
                loss = 0
                np.random.shuffle(self.dataset)
                for x, y in self.dataset:
                    loss += self.nn.train_step(x.reshape(1, -1), y)

                avg_loss = loss / len(self.dataset)
                self.nn.history.append(avg_loss)
                self.progress["value"] = i + 1
                self.root.update_idletasks()

            self.nn.save_model()
            self.progress["value"] = 0
            messagebox.showinfo(
                "OK", f"Uczenie zakonczone!\nBłąd koncowy: {self.nn.history[-1]:.4f}"
            )

        threading.Thread(target=training_process, daemon=True).start()

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
            f"LOSS {self.nn.history[-1]:.6f}",
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
        win.title("Analityka Testowa")
        win.geometry("600x650")
        win.configure(bg="#000000")

        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#000000")
        ax.set_facecolor("#000000")
        bars = ax.bar(
            labels, percents, color="#3498db", edgecolor="white", linewidth=0.5
        )

        ax.set_ylim(0, 110)
        ax.grid(True, color="#222", linestyle="--", linewidth=0.5, axis="y")
        ax.set_title(
            "DOKŁADNOŚĆ PO KLASACH (%)",
            color="white",
            fontsize=12,
            fontweight="bold",
            pad=20,
        )
        ax.tick_params(colors="white", labelsize=10)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 3,
                f"{int(height)}%",
                ha="center",
                va="bottom",
                color="#00ff41",
                fontweight="bold",
            )

        for spine in ax.spines.values():
            spine.set_color("#444")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        info_frame = tk.Frame(win, bg="#111", bd=1, relief="sunken")
        info_frame.pack(fill="x", padx=20, pady=(0, 20))

        main_info = f"OGÓLNY WYNIK: {correct} / {total} ({accuracy:.1f}%)"
        tk.Label(
            info_frame,
            text=main_info,
            fg="#00ff41",
            bg="#111",
            font=("Courier", 14, "bold"),
            pady=10,
        ).pack()

        details_str = " | ".join(
            [f"{labels[i]}: {class_correct[i]}/{class_total[i]}" for i in range(3)]
        )
        tk.Label(
            info_frame, text=details_str, fg="#888", bg="#111", font=("Arial", 10)
        ).pack(pady=(0, 10))

        # Кнопка закриття
        tk.Button(
            win,
            text="ZAMKNIJ",
            bg="#333",
            fg="white",
            command=win.destroy,
            relief="flat",
            cursor="hand2",
        ).pack(pady=5)

    def predict(self):
        img = self.process_image(self.drawing_data)
        if img is None:
            messagebox.showwarning("Błąd", "Najpierw narysuj coś na polu!")
            return

        self.nn.forward(img.flatten().reshape(1, -1))
        logits = self.nn.z3[0]
        probs = self.nn.softmax(self.nn.z3)[0]

        top_indices = np.argsort(probs)[-2:][::-1]
        idx1, idx2 = top_indices[0], top_indices[1]
        prob1, prob2 = probs[idx1], probs[idx2]

        labels = ["P", "R", "O"]

        is_trash = logits[idx1] < 0.8
        is_uncertain = prob1 < 0.65 or (prob1 - prob2) < 0.30
        ink_density = np.mean(img)
        is_weird_shape = ink_density < 0.05 or ink_density > 0.5

        if is_trash or is_uncertain or is_weird_shape:
            self.canvas.config(highlightbackground="#e74c3c")
            debug_msg = (
                f"Logit: {logits[idx1]:.2f}, Prob: {prob1:.2f}, Ink: {ink_density:.2f}"
            )
            print(f"Відхилено: {debug_msg}")
            messagebox.showwarning(
                "Nie rozpoznano",
                "To nie przypomina żadnej ze znanych liter (P, R, O).\n\n"
                "Spróbuj narysować wyraźniej.",
            )
        else:
            self.canvas.config(highlightbackground="#00ff41")
            messagebox.showinfo(
                "Sukces",
                f"Rozpoznano literę: {labels[idx1]}\nPewność: {prob1 * 100:.1f}%",
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
