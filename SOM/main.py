import tkinter as tk
import numpy as np

class SOM2D:
    def __init__(self, wielkosc=8, typ_ksztaltu="circle"):
        self.wielkosc = wielkosc
        self.typ_ksztaltu = typ_ksztaltu
        self.poczatkowe_lr = 0.6 
        self.poczatkowa_sigma = wielkosc / 2.0
        self.x, self.y = np.meshgrid(np.arange(wielkosc), np.arange(wielkosc))
        self.resetuj()

    def resetuj(self):
        self.wagi = 0.49 + 0.02 * np.random.rand(self.wielkosc, self.wielkosc, 2)
        self.lr = self.poczatkowe_lr
        self.sigma = self.poczatkowa_sigma

    def ogranicz_wagi(self):
        if self.typ_ksztaltu == "circle":
            srodek = np.array([0.5, 0.5])
            roznica = self.wagi - srodek
            odleglosc = np.linalg.norm(roznica, axis=2)
            maska = odleglosc > 0.35
            if np.any(maska):
                roznica[maska] *= (0.35 / odleglosc[maska])[..., np.newaxis]
                self.wagi[maska] = srodek + roznica[maska]

        elif self.typ_ksztaltu == "square":
            self.wagi = np.clip(self.wagi, 0.15, 0.85)

        elif self.typ_ksztaltu == "star":
            self.wagi = np.clip(self.wagi, 0.1, 0.9)

    def krok_treningowy(self, dane):
        p = dane[np.random.randint(len(dane))]
        
        odleglosci = np.linalg.norm(self.wagi - p, axis=2)
        bx, by = np.unravel_index(np.argmin(odleglosci), odleglosci.shape)
        
        odleglosc_kw = (self.x - bx)**2 + (self.y - by)**2
        wplyw = np.exp(-odleglosc_kw / (2 * self.sigma**2))
        
        self.wagi += self.lr * wplyw[..., np.newaxis] * (p - self.wagi)
        
        self.ogranicz_wagi()
        
        self.lr *= 0.9995
        self.sigma *= 0.9995

def pobierz_dane_kola(n=2000):
    r = np.sqrt(np.random.rand(n)) * 0.35
    a = np.random.rand(n) * 2 * np.pi
    return np.column_stack((0.5 + r * np.cos(a), 0.5 + r * np.sin(a)))

def pobierz_dane_kwadratu(n=2000):
    return np.random.rand(n, 2) * 0.7 + 0.15

def pobierz_dane_gwiazdy(n=3000):
    punkty = []
    while len(punkty) < n:
        p = np.random.rand(2)
        dx, dy = p[0] - 0.5, p[1] - 0.5
        r = np.sqrt(dx**2 + dy**2)
        kat = np.arctan2(dy, dx)
        r_gwiazdy = 0.15 + 0.20 * (np.abs(np.cos(kat * 2.5)))**2 
        if r < r_gwiazdy:
            punkty.append(p)
    return np.array(punkty)

class Aplikacja:
    def __init__(self, window):
        self.window = window
        self.window.title("SOM Ograniczone Wypełnianie Kształtów")
        
        panel = tk.Frame(window)
        panel.pack(pady=10)
        
        tk.Label(panel, text="Kroki:").pack(side=tk.LEFT)
        self.pole_krokow = tk.Entry(panel, width=8)
        self.pole_krokow.insert(0, "15000")
        self.pole_krokow.pack(side=tk.LEFT, padx=5)
        
        self.przycisk_start = tk.Button(panel, text="START", command=self.start, bg="#28a745", fg="white", width=10)
        self.przycisk_start.pack(side=tk.LEFT, padx=5)
        
        tk.Button(panel, text="RESET", command=self.resetuj, width=10).pack(side=tk.LEFT)

        self.ramka = tk.Frame(window)
        self.ramka.pack(padx=10, pady=10)

        self.plotna = []
        self.nazwy_ksztaltow = ["circle", "square", "star"]
        wyswietlane_nazwy = ["Koło", "Kwadrat", "Gwiazda"]
        
        for i in range(3):
            f = tk.Frame(self.ramka)
            f.pack(side=tk.LEFT, padx=5)
            tk.Label(f, text=wyswietlane_nazwy[i], font=("Arial", 10, "bold")).pack()
            c = tk.Canvas(f, width=300, height=300, bg="white", highlightthickness=1, relief="sunken")
            c.pack()
            self.plotna.append(c)

        self.resetuj()

    def resetuj(self):
        self.dziala = False
        self.sieci_som = [SOM2D(8, self.nazwy_ksztaltow[i]) for i in range(3)]
        self.dane = [pobierz_dane_kola(), pobierz_dane_kwadratu(), pobierz_dane_gwiazdy()]
        self.krok = 0
        self.rysuj()

    def rysuj_ksztalty(self, c, idx):
        srodek = 150
        skala = 300
        if idx == 0:
            c.create_oval(45, 45, 255, 255, outline="#ddd", width=2)
        elif idx == 1:
            c.create_rectangle(45, 45, 255, 255, outline="#ddd", width=2)
        elif idx == 2:
            punkty = []
            for k in range(10):
                a = k * np.pi / 5 - np.pi/2
                r = (0.35 if k % 2 == 0 else 0.15) * skala
                punkty.extend([srodek + r * np.cos(a), srodek + r * np.sin(a)])
            c.create_polygon(punkty, outline="#ddd", fill="", width=2)

    def rysuj(self):
        for idx, c in enumerate(self.plotna):
            c.delete("all")
            self.rysuj_ksztalty(c, idx)
            wagi = self.sieci_som[idx].wagi
            s = self.sieci_som[idx].wielkosc
            for i in range(s):
                for j in range(s):
                    x, y = wagi[i,j] * 300
                    if i < s - 1:
                        x2, y2 = wagi[i+1, j] * 300
                        c.create_line(x, y, x2, y2, fill="#007bff", width=1)
                    if j < s - 1:
                        x2, y2 = wagi[i, j+1] * 300
                        c.create_line(x, y, x2, y2, fill="#007bff", width=1)
            for i in range(s):
                for j in range(s):
                    x, y = wagi[i,j] * 300
                    c.create_oval(x-3, y-3, x+3, y+3, fill="#dc3545", outline="white")

    def start(self):
        if not self.dziala:
            self.maks_krokow = int(self.pole_krokow.get())
            self.dziala = True
            self.petla()

    def petla(self):
        if not self.dziala or self.krok >= self.maks_krokow:
            self.dziala = False
            return
        for _ in range(100):
            for i in range(3):
                self.sieci_som[i].krok_treningowy(self.dane[i])
            self.krok += 1
        self.rysuj()
        self.window.after(10, self.petla)

if __name__ == "__main__":
    window = tk.Tk()
    aplikacja = Aplikacja(window)
    window.mainloop()