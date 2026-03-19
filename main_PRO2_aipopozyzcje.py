import tkinter as tk
from tkinter import messagebox
import numpy as np
import os

class DeepNeuralNetwork:
    def __init__(self):
        self.input_size = 784
        self.hidden_size = 64
        self.output_size = 3
        self.lr = 0.05
        self.reset_weights()

    def reset_weights(self):
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2./self.input_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2./self.hidden_size)

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

    def train_step(self, X, y_target):
        output = self.forward(X)
        error = output - y_target
        d_W2 = self.hidden.T.dot(error)
        error_hidden = error.dot(self.W2.T) * self.leaky_relu_derivative(self.hidden)
        d_W1 = X.T.dot(error_hidden)
        self.W2 -= self.lr * d_W2
        self.W1 -= self.lr * d_W1

    def save_model(self, filename="model_weights.npz"):
        np.savez(filename, W1=self.W1, W2=self.W2)

    def load_model(self, filename="model_weights.npz"):
        if os.path.exists(filename):
            data = np.load(filename)
            self.W1 = data['W1']
            self.W2 = data['W2']
            return True
        return False

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sieć PRO - Pełny Reset")
        self.root.geometry("800x600")
        
        self.grid_size = 28
        self.cell_w = 10
        self.cell_h = 10
        
        self.nn = DeepNeuralNetwork()
        self.nn.load_model()
        
        self.db_file = "training_data.npz"
        self.weights_file = "model_weights.npz"
        self.load_database()
        
        self.drawing_data = np.zeros((self.grid_size, self.grid_size))
        self.setup_ui()

    def load_database(self):
        if os.path.exists(self.db_file):
            data = np.load(self.db_file)
            self.X_train = list(data['X'])
            self.y_train = list(data['y'])
        else:
            self.X_train, self.y_train = [], []

    def save_database(self):
        np.savez(self.db_file, X=np.array(self.X_train), y=np.array(self.y_train))

    def setup_ui(self):
        self.main_container = tk.Frame(self.root, padx=10, pady=10)
        self.main_container.pack(fill="both", expand=True)

        self.canvas_frame = tk.Frame(self.main_container)
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<Configure>", self.on_resize)

        self.ctrl_frame = tk.Frame(self.main_container, padx=10, width=150)
        self.ctrl_frame.pack(side="right", fill="y")

        tk.Label(self.ctrl_frame, text="DANE", font=("Arial", 10, "bold")).pack()
        for i, char in enumerate(["P", "R", "O"]):
            tk.Button(self.ctrl_frame, text=f"Dodaj {char}", width=15, command=lambda idx=i: self.add_sample(idx)).pack(pady=2)

        self.db_label = tk.Label(self.ctrl_frame, text=f"Baza: {len(self.y_train)}")
        self.db_label.pack(pady=10)

        tk.Label(self.ctrl_frame, text="NAUKA", font=("Arial", 10, "bold")).pack()
        self.epoch_entry = tk.Entry(self.ctrl_frame, width=10, justify="center")
        self.epoch_entry.insert(0, "100")
        self.epoch_entry.pack()
        tk.Button(self.ctrl_frame, text="Trenuj", bg="#cce5ff", command=self.run_training).pack(pady=5)

        tk.Label(self.ctrl_frame, text="TEST", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(self.ctrl_frame, text="PRZEWIDUJ", bg="#d4edda", command=self.predict).pack(fill="x")
        
        self.res_label = tk.Label(self.ctrl_frame, text="---", font=("Arial", 14, "bold"))
        self.res_label.pack(pady=10)

        tk.Label(self.ctrl_frame, text="RESETOWANIE", font=("Arial", 10, "bold"), fg="red").pack(pady=(20, 5))
        tk.Button(self.ctrl_frame, text="Wyczyść pamięć", bg="#ffcccc", command=self.full_system_reset).pack(fill="x")

        tk.Button(self.ctrl_frame, text="Wyczyść ekran", command=self.clear_canvas).pack(side="bottom", fill="x", pady=5)

    def on_resize(self, event):
        self.cell_w = event.width / self.grid_size
        self.cell_h = event.height / self.grid_size
        self.redraw_from_data()

    def draw(self, event):
        x = int(event.x / self.cell_w)
        y = int(event.y / self.cell_h)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if self.drawing_data[y, x] == 0:
                self.drawing_data[y, x] = 1.0
                self.render_cell(x, y)

    def render_cell(self, x, y):
        x1, y1 = x * self.cell_w, y * self.cell_h
        x2, y2 = (x + 1) * self.cell_w, (y + 1) * self.cell_h
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="#f0f0f0", tags="pixels")

    def redraw_from_data(self):
        self.canvas.delete("pixels")
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.drawing_data[y, x] > 0:
                    self.render_cell(x, y)

    def add_sample(self, label_idx):
        if np.sum(self.drawing_data) > 0:
            self.X_train.append(self.drawing_data.flatten().copy())
            self.y_train.append(label_idx)
            self.save_database()
            self.db_label.config(text=f"Baza: {len(self.y_train)}")
            self.clear_canvas()

    def run_training(self):
        if len(self.X_train) < 3: return
        try:
            epochs = int(self.epoch_entry.get())
        except:
            epochs = 100
        X, y = np.array(self.X_train), np.array(self.y_train)
        y_oh = np.eye(3)[y]
        for _ in range(epochs):
            for i in range(len(X)):
                self.nn.train_step(X[i:i+1], y_oh[i:i+1])
        self.nn.save_model()
        messagebox.showinfo("OK", "Wytrenowano!")

    def predict(self):
        out = self.nn.forward(self.drawing_data.flatten().reshape(1, -1))
        idx = np.argmax(out)
        self.res_label.config(text=f"{['P','R','O'][idx]} ({out[0][idx]*100:.0f}%)")

    def full_system_reset(self):
        if messagebox.askyesno("Reset", "Czy na pewno usunąć całą bazę danych i wagi sieci?"):
            if os.path.exists(self.db_file): os.remove(self.db_file)
            if os.path.exists(self.weights_file): os.remove(self.weights_file)
            
            self.X_train, self.y_train = [], []
            self.nn.reset_weights()
            self.db_label.config(text="Baza: 0")
            self.res_label.config(text="---")
            self.clear_canvas()
            messagebox.showinfo("Reset", "Pamięć została wyczyszczona.")

    def clear_canvas(self):
        self.drawing_data.fill(0)
        self.canvas.delete("pixels")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()