import tkinter as tk
from tkinter import messagebox
import numpy as np

class DeepNeuralNetwork:
    def __init__(self):
        self.input_size = 784
        self.hidden_size = 64
        self.output_size = 3
        self.lr = 0.05 
        self.reset_weights()

    def reset_weights(self):
        # Inicjalizacja wag niewielkimi wartościami losowymi dla stabilności procesu uczenia
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1

    def leaky_relu(self, x):
        # Funkcja aktywacji z lekkim nachyleniem dla wartości ujemnych, zapobiega "martwym neuronom"
        return np.where(x > 0, x, x * 0.01)

    def leaky_relu_derivative(self, x):
        # Pochodna Leaky ReLU używana do wyznaczania kierunku zmiany wag
        return np.where(x > 0, 1, 0.01)

    def softmax(self, x):
        # Normalizacja wyników do rozkładu prawdopodobieństwa (suma = 1)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Obliczanie aktywacji kolejnych warstw (przejście w przód)
        self.z1 = np.dot(X, self.W1)
        self.hidden = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.hidden, self.W2)
        self.output = self.softmax(self.z2)
        return self.output

    def train(self, X, y_label):
        target = np.zeros((1, 3))
        target[0, y_label] = 1
        output = self.forward(X)
        
        # Wsteczna propagacja błędu (Backpropagation)
        error = output - target
        
        # Obliczanie gradientów dla wag warstwy wyjściowej
        d_W2 = self.hidden.T.dot(error)
        
        # Obliczanie błędu warstwy ukrytej i gradientów wag wejściowych
        error_hidden = error.dot(self.W2.T) * self.leaky_relu_derivative(self.hidden)
        d_W1 = X.T.dot(error_hidden)
        
        # Aktualizacja parametrów sieci
        self.W2 -= self.lr * d_W2
        self.W1 -= self.lr * d_W1

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sieć Neuronowa P-R-O")
        self.root.resizable(False, False)
        self.nn = DeepNeuralNetwork()
        self.counts = [0, 0, 0]
        self.drawing_data = np.zeros((28, 28))
        self.mode = tk.StringVar(value="TEST")
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack()

        canvas_frame = tk.Frame(main_frame, highlightbackground="black", highlightthickness=2)
        canvas_frame.grid(row=0, column=0, rowspan=5)
        self.canvas = tk.Canvas(canvas_frame, width=280, height=280, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        mode_frame = tk.LabelFrame(main_frame, text="Tryb pracy", padx=5, pady=5)
        mode_frame.grid(row=0, column=1, padx=10, sticky="ew")
        tk.Radiobutton(mode_frame, text="Testowanie", variable=self.mode, value="TEST").pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Nauczanie", variable=self.mode, value="NAUKA").pack(anchor="w")

        self.btn_labels = []
        for i, char in enumerate(["P", "R", "O"]):
            frame = tk.Frame(main_frame)
            frame.grid(row=i+1, column=1, padx=10, pady=2, sticky="ew")
            btn = tk.Button(frame, text=char, width=8, bg="#e1e1e1", command=lambda idx=i: self.on_letter_btn_click(idx))
            btn.pack(side="left")
            lbl = tk.Label(frame, text="0.0%", font=('Arial', 9, 'bold'))
            lbl.pack(side="left", padx=5)
            self.btn_labels.append((btn, lbl))

        self.counter_label = tk.Label(main_frame, text="Trening: P:0 R:0 O:0", fg="blue")
        self.counter_label.grid(row=4, column=1, pady=5)

        bottom_frame = tk.Frame(self.root, pady=10)
        bottom_frame.pack()
        tk.Button(bottom_frame, text="Sprawdź", bg="#d4edda", width=10, command=self.predict).pack(side="left", padx=5)
        tk.Button(bottom_frame, text="Wyczyść", bg="#fff3cd", width=10, command=self.clear_canvas).pack(side="left", padx=5)
        tk.Button(bottom_frame, text="Reset", bg="#f8d7da", width=10, command=self.full_reset).pack(side="left", padx=5)

    def draw(self, event):
        x, y = event.x // 10, event.y // 10
        if 0 <= x < 28 and 0 <= y < 28:
            self.drawing_data[y, x] = 1.0
            self.canvas.create_rectangle(x*10, y*10, (x+1)*10, (y+1)*10, fill="black")

    def predict(self):
        if self.mode.get() == "NAUKA":
            messagebox.showwarning("Uwaga", "Przełącz na 'Testowanie'!")
            return
        
        input_vec = self.drawing_data.flatten().reshape(1, -1)
        if np.sum(input_vec) == 0:
            messagebox.showwarning("Uwaga", "Narysuj coś najpierw!")
            return
            
        outputs = self.nn.forward(input_vec)
        for i, (btn, lbl) in enumerate(self.btn_labels):
            perc = outputs[0][i] * 100
            lbl.config(text=f"{perc:.1f}%")
            btn.configure(bg="#28a745" if i == np.argmax(outputs) else "#f0f0f0")

    def on_letter_btn_click(self, idx):
        if self.mode.get() == "NAUKA":
            self.nn.train(self.drawing_data.flatten().reshape(1, -1), idx)
            self.counts[idx] += 1
            self.counter_label.config(text=f"Trening: P:{self.counts[0]} R:{self.counts[1]} O:{self.counts[2]}")
            self.clear_canvas()
        else:
            messagebox.showinfo("Info", "W trybie testu te przyciski nie uczą sieci.")

    def clear_canvas(self):
        self.drawing_data.fill(0)
        self.canvas.delete("all")
        for btn, lbl in self.btn_labels:
            btn.configure(bg="#e1e1e1")
            lbl.config(text="0.0%")

    def full_reset(self):
        if messagebox.askyesno("Reset", "Zresetować wszystko?"):
            self.nn.reset_weights()
            self.counts = [0, 0, 0]
            self.counter_label.config(text="Trening: P:0 R:0 O:0")
            self.clear_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()