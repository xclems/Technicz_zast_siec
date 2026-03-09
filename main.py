import tkinter as tk
from tkinter import messagebox
import numpy as np
class DeepNeuralNetwork:
    def __init__(self):
        self.reset_weights()

    def reset_weights(self):
        self.input_size = 784
        self.h1_size = 128
        self.h2_size = 64
        self.output_size = 3

        self.W1 = np.random.randn(self.input_size, self.h1_size) * np.sqrt(2./self.input_size)
        self.b1 = np.zeros((1, self.h1_size))
        self.W2 = np.random.randn(self.h1_size, self.h2_size) * np.sqrt(2./self.h1_size)
        self.b2 = np.zeros((1, self.h2_size))
        self.W3 = np.random.randn(self.h2_size, self.output_size) * np.sqrt(2./self.h2_size)
        self.b3 = np.zeros((1, self.output_size))
        self.lr = 0.05

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def train(self, X, y_label):
        target = np.zeros((1, 3))
        target[0, y_label] = 1
        output = self.forward(X)
        
        dz3 = output - target
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        dz2 = np.dot(dz3, self.W3.T) * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sieć Neuronowa P-R-O")
        self.root.resizable(False, False)
        self.nn = DeepNeuralNetwork()
        
        self.counts = [0, 0, 0]
        self.canvas_size = 280
        self.grid_res = 28
        self.pixel_w = self.canvas_size // self.grid_res
        self.drawing_data = np.zeros((self.grid_res, self.grid_res))

        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack()

        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, 
                                bg="white", highlightthickness=2, highlightbackground="black")
        self.canvas.grid(row=0, column=0, rowspan=6, padx=(0, 20))
        self.canvas.bind("<B1-Motion>", self.draw)

        self.btns = []
        letters = ["P", "R", "O"]
        for i, char in enumerate(letters):
            btn = tk.Button(main_frame, text=char, width=12, font=('Arial', 12, 'bold'),
                            command=lambda idx=i: self.on_letter_btn_click(idx))
            btn.grid(row=i, column=1, pady=5)
            self.btns.append(btn)

        self.counter_label = tk.Label(main_frame, text="Trening: P:0 R:0 O:0", font=('Arial', 10, 'italic'))
        self.counter_label.grid(row=3, column=1, pady=10)

        self.is_training = tk.BooleanVar()
        self.chk_train = tk.Checkbutton(main_frame, text="Tryb Nauki", variable=self.is_training, font=('Arial', 10))
        self.chk_train.grid(row=4, column=1)

        bottom_frame = tk.Frame(self.root, pady=10)
        bottom_frame.pack()

        self.btn_done = tk.Button(bottom_frame, text="Gotowe", bg="#d4edda", width=12, command=self.predict)
        self.btn_done.grid(row=0, column=0, padx=5)

        self.btn_clear = tk.Button(bottom_frame, text="Wyczyść", bg="#fff3cd", width=12, command=self.clear_canvas)
        self.btn_clear.grid(row=0, column=1, padx=5)

        self.btn_reset = tk.Button(bottom_frame, text="Reset Ogólny", bg="#f8d7da", width=12, command=self.full_reset)
        self.btn_reset.grid(row=0, column=2, padx=5)

    def draw(self, event):
        x, y = event.x // self.pixel_w, event.y // self.pixel_w
        if 0 <= x < self.grid_res and 0 <= y < self.grid_res:
            for dx in range(2): 
                for dy in range(2):
                    if x+dx < self.grid_res and y+dy < self.grid_res:
                        self.drawing_data[y+dy, x+dx] = 1.0
                        self.canvas.create_rectangle((x+dx)*self.pixel_w, (y+dy)*self.pixel_w, 
                                                    (x+dx+1)*self.pixel_w, (y+dy+1)*self.pixel_w, 
                                                    fill="black", outline="")

    def update_counters(self):
        self.counter_label.config(text=f"Trening: P:{self.counts[0]} R:{self.counts[1]} O:{self.counts[2]}")

    def on_letter_btn_click(self, idx):
        if self.is_training.get():
            input_vec = self.drawing_data.flatten().reshape(1, -1)
            if np.sum(input_vec) > 0:
                self.nn.train(input_vec, idx)
                self.counts[idx] += 1
                self.update_counters()
                self.clear_canvas()
        else:
            messagebox.showinfo("Informacja", "Włącz 'Tryb Nauki', aby trenować sieć tymi przyciskami.")

    def predict(self):
        if self.is_training.get():
            messagebox.showwarning("Uwaga", "Wyłącz tryb nauki, aby rozpoznać literę!")
            return
            
        input_vec = self.drawing_data.flatten().reshape(1, -1)
        if np.sum(input_vec) == 0: return
        
        outputs = self.nn.forward(input_vec)
        winner = np.argmax(outputs)
        
        self.reset_btn_colors()
        self.btns[winner].configure(bg="#28a745", fg="white")

    def reset_btn_colors(self):
        for btn in self.btns:
            btn.configure(bg="#f0f0f0", fg="black")

    def clear_canvas(self):
        self.drawing_data.fill(0)
        self.canvas.delete("all")
        self.reset_btn_colors()

    def full_reset(self):
        if messagebox.askyesno("Reset", "Czy na pewno chcesz zresetować wagi sieci i liczniki?"):
            self.nn.reset_weights()
            self.counts = [0, 0, 0]
            self.update_counters()
            self.clear_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()