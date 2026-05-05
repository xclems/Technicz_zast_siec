import tkinter as tk
from tkinter import ttk
import numpy as np

def circle_pts(n):
    a = np.linspace(0, 2*np.pi, n)
    return np.column_stack((0.5 + 0.35*np.cos(a),
                            0.5 + 0.35*np.sin(a)))

def square_pts(n):
    pts = []
    for i in range(n):
        t = i / n
        if t < 0.25:
            pts.append([0.15 + t*4*0.7, 0.15])
        elif t < 0.5:
            pts.append([0.85, 0.15 + (t-0.25)*4*0.7])
        elif t < 0.75:
            pts.append([0.85 - (t-0.5)*4*0.7, 0.85])
        else:
            pts.append([0.15, 0.85 - (t-0.75)*4*0.7])
    return np.array(pts)

def triangle_pts(n):
    A = np.array([0.5, 0.15])
    B = np.array([0.15, 0.85])
    C = np.array([0.85, 0.85])

    pts = []
    for i in range(n):
        t = i / n
        if t < 1/3:
            pts.append(A + (B-A)*(t*3))
        elif t < 2/3:
            pts.append(B + (C-B)*((t-1/3)*3))
        else:
            pts.append(C + (A-C)*((t-2/3)*3))
    return np.array(pts)

def diamond_pts(n):
    # Вершины ромба (верх, право, низ, лево)
    A = np.array([0.5, 0.15])
    B = np.array([0.85, 0.5])
    C = np.array([0.5, 0.85])
    D = np.array([0.15, 0.5])

    pts = []
    for i in range(n):
        t = i / n
        if t < 0.25:
            pts.append(A + (B - A) * (t * 4))          # От верхней к правой
        elif t < 0.5:
            pts.append(B + (C - B) * ((t - 0.25) * 4)) # От правой к нижней
        elif t < 0.75:
            pts.append(C + (D - C) * ((t - 0.5) * 4))  # От нижней к левой
        else:
            pts.append(D + (A - D) * ((t - 0.75) * 4)) # От левой к верхней
    return np.array(pts)

def trapezoid_pts(n):
    # Вершины трапеции
    A = np.array([0.2, 0.15]) # Верхняя левая
    B = np.array([0.8, 0.15]) # Верхняя правая
    C = np.array([0.6, 0.85]) # Нижняя правая
    D = np.array([0.4, 0.85]) # Нижняя левая

    pts = []
    for i in range(n):
        t = i / n
        if t < 0.25:
            pts.append(A + (B - A) * (t * 4))          # Верхняя грань
        elif t < 0.5:
            pts.append(B + (C - B) * ((t - 0.25) * 4)) # Правая грань
        elif t < 0.75:
            pts.append(C + (D - C) * ((t - 0.5) * 4))  # Нижняя грань
        else:
            pts.append(D + (A - D) * ((t - 0.75) * 4)) # Левая грань
    return np.array(pts)



SHAPE_DATA_FNS = {
    "Koło": circle_pts,
    "Kwadrat": square_pts,
    "Trójkąt": triangle_pts,
    "Romb": diamond_pts,
    "Trapez": trapezoid_pts,
}

# SOM
class SOM:
    def __init__(self, w=10, h=10, mode="grid"):
        self.mode = mode
        self.w, self.h = w, h
        self.n = w * h
        if mode == "grid":
            self.weights = np.random.rand(self.h, self.w, 2) * 0.05 + 0.475
        else:
            self.weights = np.random.rand(self.n, 2) * 0.05 + 0.475
        self.reset_learning_params()

    def reset_learning_params(self):
        self.eta = 0.12
        self.S = max(self.w, self.h) / 2 if self.mode == "grid" else self.n / 2

    def wake_up_for_morph(self):
        self.eta = 0.03
        self.S = max(self.w, self.h) / 4 if self.mode == "grid" else self.n / 4

    def apply_chain_elasticity(self, w):
        """
        Имитирует физическую упругость нити. 
        Плавно стягивает узлы к центру между их соседями, выравнивая расстояние
        и предотвращая любые скручивания и хаотичные переплетения.
        """
        new_w = w.copy()
        N = len(w)
        alpha = 0.08  # Коэффициент упругости (натяжения)

        for i in range(N):
            if hasattr(self, "closed") and self.closed:
                # В замкнутом режиме 0-й и последний узлы являются полноценными соседями
                left = (i - 1) % N
                right = (i + 1) % N
                avg = (w[left] + w[right]) / 2.0
                new_w[i] += alpha * (avg - w[i])
            else:
                if i == 0:
                    # Мягко удерживаем концы, чтобы они не отрывались от формы
                    new_w[i] += alpha * 0.5 * (w[1] - w[0])
                elif i == N - 1:
                    new_w[i] += alpha * 0.5 * (w[N-2] - w[N-1])
                else:
                    avg = (w[i-1] + w[i+1]) / 2.0
                    new_w[i] += alpha * (avg - w[i])
                    
        return new_w

    def step(self, p):
        self.last_input = p
        if self.mode == "grid":
            d = np.sum((self.weights - p)**2, axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)

            for y in range(self.h):
                for x in range(self.w):
                    dist2 = (i - y)**2 + (j - x)**2
                    h = np.exp(-dist2 / (2 * (self.S**2)))
                    self.weights[y, x] += self.eta * h * (p - self.weights[y, x])

            self.weights = self.smooth_grid(self.weights)

        else:
            d = np.sum((self.weights - p)**2, axis=1)
            idx = np.argmin(d)

            for i in range(len(self.weights)):
                dist = abs(i - idx)

                if hasattr(self, "closed") and self.closed:
                    # Расстояние по кольцу (кратчайшее)
                    dist = min(dist, len(self.weights) - dist)

                h = np.exp(-(dist**2)/(2*(self.S**2)))
                self.weights[i] += self.eta * h * (p - self.weights[i])

            # 🔥 Единая функция упругости, обеспечивающая ровные расстояния и форму
            self.weights = self.apply_chain_elasticity(self.weights)

        self.weights = np.clip(self.weights, 0.05, 0.95)

        self.eta *= 0.9995
        self.S *= 0.9995
    
    def smooth_grid(self, w):
        new_w = w.copy()
        for i in range(1, self.h-1):
            for j in range(1, self.w-1):
                new_w[i, j] = (
                    0.5 * w[i, j] +
                    0.125 * (w[i-1, j] + w[i+1, j] + w[i, j-1] + w[i, j+1])
                )
        return new_w


# GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title("SOM Morphing")
        root.minsize(950, 500)

        top = tk.Frame(root)
        top.pack(pady=10)

        self.left_shape_ui = ttk.Combobox(top, values=list(SHAPE_DATA_FNS.keys()), width=10, state="readonly")
        self.left_shape_ui.set("Koło")
        self.left_shape_ui.grid(row=0, column=0, padx=10)

        self.status_label = tk.Label(top, text="Status: READY")
        self.status_label.grid(row=0, column=1, padx=20)

        self.right_shape_ui = ttk.Combobox(top, values=list(SHAPE_DATA_FNS.keys()), width=10, state="readonly")
        self.right_shape_ui.set("Kwadrat")
        self.right_shape_ui.grid(row=0, column=2, padx=10)

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

        self.mode_var = tk.StringVar(value="grid")
        tk.Radiobutton(params_frame, text="Siatka", variable=self.mode_var, value="grid", command=self.on_mode_change).grid(row=0, column=4, padx=10)
        tk.Radiobutton(params_frame, text="Łańcuch", variable=self.mode_var, value="chain", command=self.on_mode_change).grid(row=0, column=5)
        
        self.closed = tk.BooleanVar()
        tk.Checkbutton(params_frame, text="Zamknięty", variable=self.closed).grid(row=0, column=6)

        btns = tk.Frame(root)
        btns.pack(pady=5)
        self.start_btn = tk.Button(btns, text="START", command=self.start_morph, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btns, text="STOP", command=self.stop_morph, width=12, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(btns, text="RESET", command=self.reset, width=12).pack(side=tk.LEFT, padx=5)

        frame = tk.Frame(root)
        frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.canvas_left = tk.Canvas(frame, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas_left.pack(side=tk.LEFT, padx=10, expand=True)
        self.canvas_mid = tk.Canvas(frame, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas_mid.pack(side=tk.LEFT, padx=10, expand=True)
        self.canvas_right = tk.Canvas(frame, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas_right.pack(side=tk.LEFT, padx=10, expand=True)

        self.left_shape_ui.bind("<<ComboboxSelected>>", lambda e: self.draw_ui_shapes())
        self.right_shape_ui.bind("<<ComboboxSelected>>", lambda e: self.draw_ui_shapes())

        self.running = False
        self.current_phase = "left"
        self.reset()

    def reset(self):
        self.running = False
        self.current_phase = "left"
        self.status_label.config(text="Status: READY")
        try:
            w, h = int(self.entry_w.get()), int(self.entry_h.get())
        except: w, h = 10, 10
        self.som = SOM(w, h, self.mode_var.get())
        self.start_btn.config(state="normal", text="START")
        self.stop_btn.config(state="disabled")
        self.som.closed = self.closed.get()
        self.draw_ui_shapes()
        self.draw_som()

    def on_mode_change(self):
        new_mode = self.mode_var.get()
        if hasattr(self, 'som') and self.som.mode != new_mode:
            self.som.mode = new_mode
            if new_mode == "grid":
                self.som.weights = self.som.weights.reshape((self.som.h, self.som.w, 2))
            else:
                self.som.weights = self.som.weights.reshape((self.som.n, 2))
            self.draw_som()

    def draw_ui_shapes(self):
        self.draw_outline(self.canvas_left, self.left_shape_ui.get())
        self.draw_outline(self.canvas_right, self.right_shape_ui.get())
        self.left_data = SHAPE_DATA_FNS[self.left_shape_ui.get()](2000)
        self.right_data = SHAPE_DATA_FNS[self.right_shape_ui.get()](2000)

    def draw_outline(self, canvas, shape_name):
        canvas.delete("all")
        s = 300
        if shape_name == "Koło": canvas.create_oval(s*0.15, s*0.15, s*0.85, s*0.85, outline="blue", width=2)
        elif shape_name == "Kwadrat": canvas.create_rectangle(s*0.15, s*0.15, s*0.85, s*0.85, outline="blue", width=2)
        elif shape_name == "Trójkąt": canvas.create_polygon(s*0.5, s*0.15, s*0.15, s*0.85, s*0.85, s*0.85, fill="", outline="blue", width=2)
        elif shape_name == "Romb": canvas.create_polygon(s*0.5, s*0.15, s*0.85, s*0.5, s*0.5, s*0.85, s*0.15, s*0.5, fill="", outline="blue", width=2)
        elif shape_name == "Trapez": canvas.create_polygon(s*0.2, s*0.15, s*0.8, s*0.15, s*0.6, s*0.85, s*0.4, s*0.85, fill="", outline="blue", width=2)
        elif shape_name == "Gwiazda":
            pts = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi/2
                r = 0.35 * s if i % 2 == 0 else 0.15 * s
                pts.extend([s*0.5 + r*np.cos(angle), s*0.5 + r*np.sin(angle)])
            canvas.create_polygon(pts, fill="", outline="blue", width=2)

    def draw_som(self):
        c, s = self.canvas_mid, 300
        c.delete("all")
        w = self.som.weights
        if self.som.mode == "grid":
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    x, y = w[i,j]*s
                    if i+1 < w.shape[0]: c.create_line(x, y, *w[i+1,j]*s, fill="gray")
                    if j+1 < w.shape[1]: c.create_line(x, y, *w[i,j+1]*s, fill="gray")
        else:
            for i in range(len(w)-1):
                c.create_line(*w[i]*s, *w[i+1]*s, fill="gray")
            if self.closed.get(): c.create_line(*w[-1]*s, *w[0]*s, fill="gray")
        for p in w.reshape(-1,2):
            x, y = p*s
            c.create_oval(x-2, y-2, x+2, y+2, fill="red", outline="maroon")

    def loop(self):
        if not self.running: return
        data = self.left_data if self.current_phase == "left" else self.right_data
        self.status_label.config(text=f"Status: {self.current_phase.upper()}")
        
        # Обновляем состояние 'closed' динамически на случай если чекбокс изменился
        self.som.closed = self.closed.get()
        
        for _ in range(30):
            p = data[np.random.randint(len(data))]
            self.som.step(p)
        self.draw_som()
        
        if self.som.eta < 0.008:
            self.current_phase = "right" if self.current_phase == "left" else "left"
            self.som.wake_up_for_morph()
            self.root.after(300, self.loop)
        else:
            self.root.after(10, self.loop)

    def start_morph(self):
        self.running = True
        self.som.closed = self.closed.get()
        self.start_btn.config(state="disabled", text="RUNNING")
        self.stop_btn.config(state="normal", text="STOP")
        self.loop()

    def stop_morph(self):
        self.running = False
        self.start_btn.config(state="normal", text="RESUME")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Status: PAUSED")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()