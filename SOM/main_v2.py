import tkinter as tk
from tkinter import ttk
import numpy as np
import time

# === GENEROWANIE DANYCH (KSZTAŁTY) ===

def circle_pts(n):
    r = np.sqrt(np.random.rand(n)) * 0.35
    a = np.random.rand(n) * 2*np.pi
    return np.column_stack((0.5 + r*np.cos(a), 0.5 + r*np.sin(a)))

def square_pts(n):
    return np.random.rand(n, 2) * 0.7 + 0.15

def triangle_pts(n):
    A, B, C = np.array([0.5, 0.15]), np.array([0.15, 0.85]), np.array([0.85, 0.85])
    pts = []
    for _ in range(n):
        r1, r2 = np.random.rand(2)
        s = np.sqrt(r1)
        pts.append((1-s)*A + s*(1-r2)*B + s*r2*C)
    return np.array(pts)

def diamond_pts(n):
    pts = []
    while len(pts) < n:
        x, y = np.random.rand(2)
        if abs(x-0.5)+abs(y-0.5) < 0.35:
            pts.append([x,y])
    return np.array(pts)

def trapezoid_pts(n):
    y = np.random.rand(n)
    x_min, x_max = 0.2 + 0.2*y, 0.8 - 0.2*y
    x = x_min + (x_max - x_min) * np.random.rand(n)
    y = 0.15 + y * 0.7
    return np.column_stack((x, y))

def star_pts(n):
    pts = []
    while len(pts) < n:
        a = np.random.rand()*2*np.pi
        r_max = 0.15 + 0.2 * (np.abs(np.sin(2.5 * a)))
        r = np.sqrt(np.random.rand()) * r_max
        pts.append([0.5 + r*np.cos(a), 0.5 + r*np.sin(a)])
    return np.array(pts)

SHAPE_DATA_FNS = {
    "Koło": circle_pts, 
    "Kwadrat": square_pts, 
    "Trójkąt": triangle_pts,
    "Romb": diamond_pts, 
    "Trapez": trapezoid_pts, 
    "Gwiazda": star_pts
}

# === GŁÓWNY ALGORYTM SOM ===

class SOM:
    def __init__(self, w=10, h=10, mode="grid"):
        self.mode = mode
        self.w = w
        self.h = h
        
        if mode == "grid":
            self.weights = np.random.rand(self.h, self.w, 2)
        else:
            self.n = w * h
            self.weights = np.random.rand(self.n, 2)
            
        self.reset_learning_params()

    def reset_learning_params(self):
        self.eta = 0.1
        if self.mode == "grid":
            self.S = max(self.w, self.h) / 2
        else:
            self.S = self.n / 2

    def wake_up_for_morph(self):
        self.eta = 0.05
        if self.mode == "grid":
            self.S = max(self.w, self.h) / 4
        else:
            self.S = self.n / 4

    def step(self, p):
        if self.mode == "grid":
            d = np.sum((self.weights - p)**2, axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            for y in range(self.h):
                for x in range(self.w):
                    dist = np.sqrt((i-y)**2 + (j-x)**2)
                    if dist < self.S:
                        self.weights[y,x] += self.eta * (p - self.weights[y,x])
        else:
            d = np.sum((self.weights - p)**2, axis=1)
            idx = np.argmin(d)
            for i in range(len(self.weights)):
                dist = abs(i - idx)
                if dist < self.S:
                    self.weights[i] += self.eta * (p - self.weights[i])
        
        self.eta *= 0.9998
        self.S *= 0.9998

# === INTERFEJS UŻYTKOWNIKA (GUI) ===

class App:
    def __init__(self, root):
        self.root = root
        root.title("SOM Morphing")
        root.minsize(950, 450)
        
        top = tk.Frame(root)
        top.pack(pady=10)

        self.left_shape = ttk.Combobox(top, values=list(SHAPE_DATA_FNS.keys()), width=10, state="readonly")
        self.left_shape.set("Koło")
        self.left_shape.grid(row=0, column=0, padx=10)

        self.target_var = tk.StringVar(value="left")

        tk.Radiobutton(top, text="<- Ucz lewy", variable=self.target_var, value="left", command=self.on_target_change).grid(row=0, column=1)
        tk.Radiobutton(top, text="Ucz prawy ->", variable=self.target_var, value="right", command=self.on_target_change).grid(row=0, column=2)

        self.right_shape = ttk.Combobox(top, values=list(SHAPE_DATA_FNS.keys()), width=10, state="readonly")
        self.right_shape.set("Kwadrat")
        self.right_shape.grid(row=0, column=3, padx=10)

        params_frame = tk.Frame(root)
        params_frame.pack(pady=5)
        
        tk.Label(params_frame, text="Szerokość:").grid(row=0, column=0)
        self.entry_w = tk.Entry(params_frame, width=5)
        self.entry_w.insert(0, "10")
        self.entry_w.grid(row=0, column=1, padx=5)

        tk.Label(params_frame, text="Wysokość:").grid(row=0, column=2)
        self.entry_h = tk.Entry(params_frame, width=5)
        self.entry_h.insert(0, "10")
        self.entry_h.grid(row=0, column=3, padx=5)

        self.mode = tk.StringVar(value="grid")
        tk.Radiobutton(params_frame, text="Siatka", variable=self.mode, value="grid").grid(row=0, column=4, padx=10)
        tk.Radiobutton(params_frame, text="Łańcuch", variable=self.mode, value="chain").grid(row=0, column=5)
        
        self.closed = tk.BooleanVar()
        tk.Checkbutton(params_frame, text="Zamknięty łańcuch", variable=self.closed).grid(row=0, column=6)

        btns = tk.Frame(root)
        btns.pack(pady=5)
        self.start_btn = tk.Button(btns, text="START", command=self.start_morph, bg="green", fg="white", width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="RESET", command=self.reset, width=15).pack(side=tk.LEFT, padx=5)

        frame = tk.Frame(root)
        frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.canvas_left = tk.Canvas(frame, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas_left.pack(side=tk.LEFT, padx=10, expand=True)
        self.canvas_mid = tk.Canvas(frame, width=300, height=300, bg="#f0f0f0", highlightthickness=1)
        self.canvas_mid.pack(side=tk.LEFT, padx=10, expand=True)
        self.canvas_right = tk.Canvas(frame, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas_right.pack(side=tk.LEFT, padx=10, expand=True)

        self.left_shape.bind("<<ComboboxSelected>>", lambda e: self.draw_ui_shapes())
        self.right_shape.bind("<<ComboboxSelected>>", lambda e: self.draw_ui_shapes())

        self.running = False
        self.has_started = False
        self.reset()

    def reset(self):
        self.running = False
        self.has_started = False
        
        try:
            w = int(self.entry_w.get())
            h = int(self.entry_h.get())
            if w < 2: w = 2
            if h < 2: h = 2
        except ValueError:
            w, h = 10, 10
            self.entry_w.delete(0, tk.END)
            self.entry_w.insert(0, "10")
            self.entry_h.delete(0, tk.END)
            self.entry_h.insert(0, "10")

        self.som = SOM(w, h, self.mode.get())
        self.start_btn.config(state="normal", text="START")
        self.draw_ui_shapes()
        self.draw_som()

    def draw_outline(self, canvas, shape_name):
        canvas.delete("all")
        s = 300
        
        if shape_name == "Koło":
            canvas.create_oval(s*0.15, s*0.15, s*0.85, s*0.85, outline="blue", width=2)
        elif shape_name == "Kwadrat":
            canvas.create_rectangle(s*0.15, s*0.15, s*0.85, s*0.85, outline="blue", width=2)
        elif shape_name == "Trójkąt":
            canvas.create_polygon(s*0.5, s*0.15, s*0.15, s*0.85, s*0.85, s*0.85, fill="", outline="blue", width=2)
        elif shape_name == "Romb":
            canvas.create_polygon(s*0.5, s*0.15, s*0.85, s*0.5, s*0.5, s*0.85, s*0.15, s*0.5, fill="", outline="blue", width=2)
        elif shape_name == "Trapez":
            canvas.create_polygon(s*0.2, s*0.15, s*0.8, s*0.15, s*0.6, s*0.85, s*0.4, s*0.85, fill="", outline="blue", width=2)
        elif shape_name == "Gwiazda":
            pts = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi/2
                r = 0.35 * s if i % 2 == 0 else 0.15 * s
                pts.extend([s*0.5 + r*np.cos(angle), s*0.5 + r*np.sin(angle)])
            canvas.create_polygon(pts, fill="", outline="blue", width=2)

    def draw_ui_shapes(self):
        self.draw_outline(self.canvas_left, self.left_shape.get())
        self.draw_outline(self.canvas_right, self.right_shape.get())
        self.left_data = SHAPE_DATA_FNS[self.left_shape.get()](2000)
        self.right_data = SHAPE_DATA_FNS[self.right_shape.get()](2000)
        
        if self.has_started:
            self.on_target_change()

    def draw_som(self):
        c = self.canvas_mid
        c.delete("all")
        s = 300
        if self.mode.get() == "grid":
            w = self.som.weights
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    x, y = w[i,j]*s
                    if i+1 < w.shape[0]:
                        x2, y2 = w[i+1,j]*s
                        c.create_line(x, y, x2, y2, fill="gray")
                    if j+1 < w.shape[1]:
                        x2, y2 = w[i,j+1]*s
                        c.create_line(x, y, x2, y2, fill="gray")
        else:
            w = self.som.weights
            for i in range(len(w)-1):
                x, y, x2, y2 = *w[i]*s, *w[i+1]*s
                c.create_line(x, y, x2, y2, fill="gray")
            if self.closed.get():
                x, y, x2, y2 = *w[-1]*s, *w[0]*s
                c.create_line(x, y, x2, y2, fill="gray")

        for p in self.som.weights.reshape(-1,2):
            x, y = p*s
            c.create_oval(x-2, y-2, x+2, y+2, fill="red", outline="maroon")

    # === PĘTLA ANIMACJI ===

    def on_target_change(self):
        if not self.has_started: return
        self.som.wake_up_for_morph()
        if not self.running:
            self.running = True
            self.loop()

    def loop(self):
        if not self.running:
            return

        current_data = self.left_data if self.target_var.get() == "left" else self.right_data

        for _ in range(50):
            p = current_data[np.random.randint(len(current_data))]
            self.som.step(p)
        self.draw_som()
        if self.som.eta > 0.005:
            self.root.after(10, self.loop)
        else:
            self.running = False 

    def start_morph(self):
        if self.has_started: return
        self.som.reset_learning_params()
        self.has_started = True
        self.running = True
        self.start_btn.config(state="disabled", text="W TOKU...")
        self.loop()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()