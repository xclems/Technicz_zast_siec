import tkinter as tk

import numpy as np


# =========================
# Hopfield Network (NumPy)
# =========================
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        self.W = np.zeros((self.size, self.size))
        for p in patterns:
            p = p.reshape(-1, 1)
            self.W += p @ p.T
        np.fill_diagonal(self.W, 0)

    def predict(self, pattern, steps=5):
        x = pattern.copy()
        for _ in range(steps):
            x = np.sign(self.W @ x)
            x[x == 0] = 1
        return x


# =========================
# UI App
# =========================
GRID_SIZE = 7  # 7x7 = 49 neurons
CELL_SIZE = 40


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hopfield Network Demo")

        self.canvas = tk.Canvas(
            root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE
        )
        self.canvas.pack()

        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.rects = []

        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill="white", outline="black"
                )
                row.append(rect)
            self.rects.append(row)

        self.canvas.bind("<Button-1>", self.toggle_cell)

        self.patterns = []
        self.net = HopfieldNetwork(GRID_SIZE * GRID_SIZE)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Save Pattern", command=self.save_pattern).pack(
            side=tk.LEFT
        )
        tk.Button(btn_frame, text="Train", command=self.train).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Recall", command=self.recall).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Reset Patterns", command=self.reset_patterns).pack(
            side=tk.LEFT
        )

    def toggle_cell(self, event):
        j = event.x // CELL_SIZE
        i = event.y // CELL_SIZE
        if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
            self.grid[i, j] *= -1
            color = "black" if self.grid[i, j] == -1 else "white"
            self.canvas.itemconfig(self.rects[i][j], fill=color)

    def save_pattern(self):
        if len(self.patterns) >= 2:
            print("Already have 2 patterns")
            return
        self.patterns.append(self.grid.flatten().copy())
        print(f"Saved pattern {len(self.patterns)}")

    def train(self):
        if len(self.patterns) < 2:
            print("Need 2 patterns")
            return
        self.net.train(self.patterns)
        print("Trained")

    def recall(self):
        pattern = self.grid.flatten()
        result = self.net.predict(pattern)
        self.grid = result.reshape((GRID_SIZE, GRID_SIZE))
        self.update_ui()
        print("Recalled")

    def clear(self):
        self.grid = np.ones((GRID_SIZE, GRID_SIZE))
        self.update_ui()

    def reset_patterns(self):
        self.patterns = []
        self.net = HopfieldNetwork(GRID_SIZE * GRID_SIZE)
        print("Patterns reset")

    def update_ui(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color = "black" if self.grid[i, j] == -1 else "white"
                self.canvas.itemconfig(self.rects[i][j], fill=color)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
