import tkinter as tk
import numpy as np


class SOM2D:
    def __init__(self, w, h, typ_ksztaltu="circle"):
        self.wielkosc_x = w
        self.wielkosc_y = h
        self.typ_ksztaltu = typ_ksztaltu
        

        self.eta_start = 0.1
        self.epsEta = 0.9999
        self.S_start = np.sqrt(w * h) / 1.5 
        self.epsS = 0.9999
        
        self.resetuj()

    def resetuj(self):

        self.wagi = 0.495 + np.random.rand(self.wielkosc_y, self.wielkosc_x, 2) * 0.01
        self.eta = self.eta_start
        self.S = self.S_start

    def dist2(self, waga, wejscie):
        return (waga[0] - wejscie[0])**2 + (waga[1] - wejscie[1])**2

    def fS(self, d):
        if self.S < 0.001: return 0.0
        return 1.0 - d / self.S

    def ogranicz_wagi(self):

        h_sieci, w_sieci, _ = self.wagi.shape
        for i in range(h_sieci):
            for j in range(w_sieci):
                v = self.wagi[i, j]
                
                if self.typ_ksztaltu == "circle":
                    srodek = np.array([0.5, 0.5])
                    roznica = v - srodek
                    odl = np.linalg.norm(roznica)
                    if odl > 0.35: # Promień konturu
                        self.wagi[i, j] = srodek + roznica * (0.35 / odl)
                
                elif self.typ_ksztaltu == "square":

                    self.wagi[i, j] = np.clip(v, 0.15, 0.85)
                    
                elif self.typ_ksztaltu == "triangle":

                    v[0] = np.clip(v[0], 0.15, 0.85)
                    v[1] = np.clip(v[1], 0.15, 0.85)

    def krok_treningowy(self, dane):
        if len(dane) == 0: return
        p = dane[np.random.randint(len(dane))]
        

        dists = np.sum((self.wagi - p)**2, axis=2)
        idxW, idxK = np.unravel_index(np.argmin(dists), dists.shape)
        

        SS = int(self.S)
        for i in range(idxW - SS, idxW + SS + 1):
            if 0 <= i < self.wielkosc_y:
                for j in range(idxK - SS, idxK + SS + 1):
                    if 0 <= j < self.wielkosc_x:
                        d_siatki = np.sqrt((idxW - i)**2 + (idxK - j)**2)
                        if d_siatki < self.S:
                            wspolczynnik = self.eta * self.fS(d_siatki)
                            self.wagi[i, j] += wspolczynnik * (p - self.wagi[i, j])
        

        self.ogranicz_wagi()
        

        self.eta *= self.epsEta
        self.S *= self.epsS


def pobierz_dane_kola(n=2500):
    r = np.sqrt(np.random.rand(n)) * 0.35
    a = np.random.rand(n) * 2 * np.pi
    return np.column_stack((0.5 + r * np.cos(a), 0.5 + r * np.sin(a)))

def pobierz_dane_kwadratu(n=2500):
    return np.random.rand(n, 2) * 0.7 + 0.15

def pobierz_dane_trojkata(n=2500):

    A = np.array([0.5, 0.15])
    B = np.array([0.15, 0.85])
    C = np.array([0.85, 0.85])
    
    punkty = []
    while len(punkty) < n:

        r1, r2 = np.random.rand(2)
        sqrt_r1 = np.sqrt(r1)
        

        P = (1 - sqrt_r1) * A + (sqrt_r1 * (1 - r2)) * B + (sqrt_r1 * r2) * C
        punkty.append(P)
        
    return np.array(punkty)


class Aplikacja:
    def __init__(self, korzen):
        self.korzen = korzen
        self.korzen.title("SOM Java Logic: Circle, Square, Triangle")
        
        panel = tk.Frame(korzen)
        panel.pack(pady=10)
        
        tk.Label(panel, text="Kroki (Epochs):").pack(side=tk.LEFT)
        self.pole_krokow = tk.Entry(panel, width=10)
        self.pole_krokow.insert(0, "25000")
        self.pole_krokow.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(panel, text="START", command=self.start, bg="#28a745", fg="white", font=("Arial", 10, "bold"))
        self.btn_start.pack(side=tk.LEFT, padx=10)
        tk.Button(panel, text="RESET", command=self.resetuj).pack(side=tk.LEFT)

        self.ramka = tk.Frame(korzen)
        self.ramka.pack(padx=10, pady=10)

        self.plotna = []

        self.typy = ["circle", "square", "triangle"]
        etykiety = ["Koło", "Kwadrat", "Trójkąt"]
        
        for i in range(3):
            f = tk.Frame(self.ramka, padx=5)
            f.pack(side=tk.LEFT)
            tk.Label(f, text=etykiety[i], font=("Arial", 10, "bold")).pack(pady=5)
            c = tk.Canvas(f, width=350, height=350, bg="white", borderwidth=1, relief="solid")
            c.pack()
            self.plotna.append(c)

        self.resetuj()

    def resetuj(self):
        self.dziala = False
        self.sieci = [SOM2D(8, 8, self.typy[i]) for i in range(3)]

        self.dane = [pobierz_dane_kola(), pobierz_dane_kwadratu(), pobierz_dane_trojkata()]
        self.krok = 0
        self.rysuj()

    def rysuj_kontury(self, c, idx):
        cntr, sz = 175, 350 
        
        if idx == 0: 
            r = 0.35 * sz
            c.create_oval(cntr - r, cntr - r, cntr + r, cntr + r, outline="#eee", width=3)
        elif idx == 1: 
            r = 0.35 * sz
            c.create_rectangle(cntr - r, cntr - r, cntr + r, cntr + r, outline="#eee", width=3)
        elif idx == 2: 
            punkty_trojkata = [
                0.5 * sz, 0.15 * sz, # A
                0.15 * sz, 0.85 * sz, # B
                0.85 * sz, 0.85 * sz  # C
            ]
            c.create_polygon(punkty_trojkata, outline="#eee", fill="", width=3)

    def rysuj(self):
        for idx, c in enumerate(self.plotna):
            c.delete("all")
            

            self.rysuj_kontury(c, idx)
            
            wagi = self.sieci[idx].wagi
            h, w, _ = wagi.shape
            scale = 350
            

            for i in range(h):
                for j in range(w):
                    x, y = wagi[i, j] * scale
                    if i + 1 < h:
                        x2, y2 = wagi[i+1, j] * scale
                        c.create_line(x, y, x2, y2, fill="#007bff", width=1)
                    if j + 1 < w:
                        x2, y2 = wagi[i, j+1] * scale
                        c.create_line(x, y, x2, y2, fill="#007bff", width=1)
            
  
            for i in range(h):
                for j in range(w):
                    x, y = wagi[i, j] * scale
                    c.create_oval(x-3, y-3, x+3, y+3, fill="#dc3545", outline="white")

    def start(self):
        if not self.dziala:
            try:
                self.maks_krokow = int(self.pole_krokow.get())
                self.dziala = True
                self.btn_start.config(text="RUNNING...", bg="#ffc107", state="disabled")
                self.petla()
            except ValueError: pass

    def petla(self):
        if not self.dziala or self.krok >= self.maks_krokow:
            self.dziala = False
            self.btn_start.config(text="START", bg="#28a745", state="normal")
            return
        

        for _ in range(150): 
            for i in range(3):
                self.sieci[i].krok_treningowy(self.dane[i])
            self.krok += 1
            
        self.rysuj()
        self.korzen.after(1, self.petla)

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplikacja(root)
    root.mainloop()