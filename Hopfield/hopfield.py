import random
import tkinter as tk
from tkinter import messagebox
import time

import numpy as np


WZORZEC_P = np.array(
    [
       [-1, -1, -1, 1, 1, 1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
)

WZORZEC_R = np.array(
    [
        [-1, -1, -1, 1, 1, 1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1, 1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1, -1, 1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
)

WZORZEC_O = np.array(
    [
        [-1, -1, -1, 1, 1, 1, -1, -1, -1, -1],
    [-1, -1, 1, -1, -1, -1, 1, -1, -1, -1],
    [-1, -1, 1, -1, -1, -1, 1, -1, -1, -1],
    [-1, -1, 1, -1, -1, -1, 1, -1, -1, -1],
    [-1, -1, 1, -1, -1, -1, 1, -1, -1, -1],
    [-1, -1, 1, -1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
)



class SiecHopfielda:
    def __init__(self, rozmiar):
        self.rozmiar = rozmiar
        self.W = np.zeros((rozmiar, rozmiar))
        self.nauczona = False

    def naucz(self, wzorce):
        n = self.rozmiar
        p = len(wzorce)
        M = np.array(wzorce)
        G = M @ M.T
        G_inv = np.linalg.inv(G)
        self.W = M.T @ G_inv @ M
        np.fill_diagonal(self.W, 0)
        self.nauczona = True

    def przypomnij(self, wzorzec, kroki=40):
        x = wzorzec.copy().astype(float)
        n = self.rozmiar
        for _ in range(kroki):
            kolejnosc = np.random.permutation(n)
            stary_x = x.copy()
            for i in kolejnosc:
                s = np.dot(self.W[i], x) + (random.random() * 0.02 - 0.01)
                x[i] = 1.0 if s >= 0 else -1.0
            if np.array_equal(x, stary_x):
                break
        return x



ROZMIAR_SIATKI = 10
ROZMIAR_KOMORKI = 40
MAKS_WZORCOW = 3

KOL_TLO = "#1c1c1e"
KOL_PANEL = "#111113"
KOL_KOMORKA = "#2a2a2c"
KOL_AKTYWNA = "#e8e8e8"
KOL_LINIA = "#3a3a3c"
KOL_TEKST = "#c8c8c8"
KOL_MUTNY = "#666666"
KOL_AKCENT = "#4a90e2"
KOL_ZIELONY = "#3d9e6e"
KOL_CZERWONY = "#e24a4a"
KOL_BTN = "#2a2a2c"
KOL_BTN_HOV = "#3a3a3c"


class AplikacjaHopfield:
    def __init__(self, root):
        self.root = root
        self.root.title("Sieć Hopfielda 7x7")
        self.root.configure(bg=KOL_TLO)
        self.root.resizable(True, True)

        self.siec = SiecHopfielda(ROZMIAR_SIATKI * ROZMIAR_SIATKI)
        self.wzorce = []
        self.siatka = np.full((ROZMIAR_SIATKI, ROZMIAR_SIATKI), -1.0)
        self.rysowanie = False

        self._zbuduj_ui()
        self._odrysuj_siatke()
        self._zaladuj_wzorce_domyslne()


    def _zbuduj_ui(self):
        glowny = tk.Frame(self.root, bg=KOL_TLO)
        glowny.pack(fill=tk.BOTH, expand=True)


        lewa = tk.Frame(glowny, bg=KOL_TLO, padx=24, pady=24)
        lewa.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


        tk.Label(
            lewa,
            text="SIEĆ HOPFIELDA",
            font=("Courier", 11, "bold"),
            bg=KOL_TLO,
            fg=KOL_AKCENT,
        ).pack(anchor="w", pady=(0, 12))


        wym = ROZMIAR_SIATKI * ROZMIAR_KOMORKI
        self.kanwa = tk.Canvas(
            lewa,
            width=wym,
            height=wym,
            bg=KOL_TLO,
            highlightthickness=1,
            highlightbackground=KOL_LINIA,
            cursor="crosshair",
        )
        self.kanwa.pack(fill=tk.BOTH, expand=True)


        self.komorki = []
        for i in range(ROZMIAR_SIATKI):
            rzad = []
            for j in range(ROZMIAR_SIATKI):
                x1 = j * ROZMIAR_KOMORKI + 1
                y1 = i * ROZMIAR_KOMORKI + 1
                x2 = x1 + ROZMIAR_KOMORKI - 2
                y2 = y1 + ROZMIAR_KOMORKI - 2
                r = self.kanwa.create_rectangle(
                    x1, y1, x2, y2, fill=KOL_KOMORKA, outline=KOL_LINIA
                )
                rzad.append(r)
            self.komorki.append(rzad)

        self.kanwa.bind("<ButtonPress-1>", self._start_rysowania)
        self.kanwa.bind("<B1-Motion>", self._rysuj)
        self.kanwa.bind("<ButtonRelease-1>", self._stop_rysowania)

        self.kanwa.bind("<ButtonPress-3>", self._start_usuwania)
        self.kanwa.bind("<B3-Motion>", self._usun)
        self.kanwa.bind("<ButtonRelease-3>", self._stop_rysowania)

        self.etk_status = tk.Label(
            lewa, text="Gotowy.", font=("Courier", 10), bg=KOL_TLO, fg=KOL_MUTNY
        )
        self.etk_status.pack(pady=(10, 8))


        ramka_btn = tk.Frame(lewa, bg=KOL_TLO)
        ramka_btn.pack()

        self.btn_zapisz = self._przycisk(
            ramka_btn, "Zapisz wzorzec", self._zapisz_wzorzec
        )
        self.btn_trenuj = self._przycisk(ramka_btn, "Trenuj sieć", self._trenuj)
        self.btn_szum = self._przycisk(ramka_btn, "Dodaj szum (20%)", self._dodaj_szum)
        self.btn_rozpoznaj = self._przycisk(ramka_btn, "Odtwórz", self._rozpoznaj)
        self.btn_wyczysc = self._przycisk(
            ramka_btn,
            "Wyczyść siatkę",
            self._wyczysc
        )

        self.btn_reset = tk.Button(
            ramka_btn,
            text="Resetuj wzorce",
            command=self._resetuj_wzorce,
            font=("Courier", 10),
            bg="#7a1f1f",
            fg="#ffffff",
            activebackground="#a62d2d",
            activeforeground="#ffffff",
            relief="flat",
            width=16,
            pady=6,
            cursor="hand2",
        )

        self.btn_reset.bind(
            "<Enter>",
            lambda e: self.btn_reset.config(bg="#a62d2d")
        )

        self.btn_reset.bind(
            "<Leave>",
            lambda e: self.btn_reset.config(bg="#7a1f1f")
        )

        self.btn_zapisz.grid(row=0, column=0, padx=4, pady=3)
        self.btn_trenuj.grid(row=0, column=1, padx=4, pady=3)
        self.btn_szum.grid(row=1, column=0, padx=4, pady=3)
        self.btn_rozpoznaj.grid(row=1, column=1, padx=4, pady=3)
        self.btn_wyczysc.grid(row=2, column=0, padx=4, pady=3)
        self.btn_reset.grid(row=2, column=1, padx=4, pady=3)


        prawy = tk.Frame(glowny, bg=KOL_PANEL, width=220, padx=20, pady=24)
        prawy.pack(side=tk.RIGHT, fill=tk.Y)
        prawy.pack_propagate(False)

        tk.Label(
            prawy,
            text="INSTRUKCJA",
            font=("Courier", 10, "bold"),
            bg=KOL_PANEL,
            fg=KOL_AKCENT,
        ).pack(anchor="w", pady=(0, 14))

        instrukacja = (
            "MYSZ:\n"
            "  LPM  — rysuj\n"
            "  PPM  — gumka\n\n"
            "KROKI:\n"
            "  1. Trenuj sieć\n"
            "  2. Narysuj literę\n"
            "     lub dodaj szum\n"
            "  3. Rozpoznaj\n\n"
            "WZORCE:\n"
            "  P, R, O załadowane\n"
            "  przy starcie."
        )
        tk.Label(
            prawy,
            text=instrukacja,
            font=("Courier", 10),
            justify=tk.LEFT,
            bg=KOL_PANEL,
            fg="#999999",
        ).pack(anchor="w")


        self.etk_wzorce = tk.Label(
            prawy,
            text=f"Wzorce: 0 / {MAKS_WZORCOW}",
            font=("Courier", 10, "bold"),
            bg=KOL_PANEL,
            fg=KOL_TEKST,
        )
        self.etk_wzorce.pack(pady=(20, 6))


        self.etk_siec = tk.Label(
            prawy,
            text="Sieć: nienauczona",
            font=("Courier", 10),
            bg=KOL_PANEL,
            fg=KOL_CZERWONY,
        )
        self.etk_siec.pack()


        tk.Label(
            prawy,
            text="Zapisane wzorce:",
            font=("Courier", 9),
            bg=KOL_PANEL,
            fg=KOL_MUTNY,
        ).pack(anchor="w", pady=(18, 6))

        self.ramka_miniatur = tk.Frame(prawy, bg=KOL_PANEL)
        self.ramka_miniatur.pack(anchor="w")
        self.root.bind("<Configure>", self._resize)


    def _przycisk(self, rodzic, tekst, komenda):
        btn = tk.Button(
            rodzic,
            text=tekst,
            command=komenda,
            font=("Courier", 10),
            bg=KOL_BTN,
            fg=KOL_TEKST,
            activebackground=KOL_BTN_HOV,
            activeforeground="#ffffff",
            relief="flat",
            width=16,
            pady=6,
            cursor="hand2",
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=KOL_BTN_HOV))
        btn.bind("<Leave>", lambda e: btn.config(bg=KOL_BTN))
        return btn


    def _komorka_z_eventu(self, event):
        j = event.x // ROZMIAR_KOMORKI
        i = event.y // ROZMIAR_KOMORKI
        if 0 <= i < ROZMIAR_SIATKI and 0 <= j < ROZMIAR_SIATKI:
            return i, j
        return None

    def _start_rysowania(self, event):
        self.rysowanie = True
        kom = self._komorka_z_eventu(event)
        if kom:
            self.siatka[kom[0], kom[1]] = 1.0
            self._odrysuj_siatke()

    def _rysuj(self, event):
        if not self.rysowanie:
            return
        kom = self._komorka_z_eventu(event)
        if kom:
            self.siatka[kom[0], kom[1]] = 1.0
            self._odrysuj_siatke()

    def _start_usuwania(self, event):
        self.rysowanie = True
        kom = self._komorka_z_eventu(event)
        if kom:
            self.siatka[kom[0], kom[1]] = -1.0
            self._odrysuj_siatke()

    def _usun(self, event):
        if not self.rysowanie:
            return
        kom = self._komorka_z_eventu(event)
        if kom:
            self.siatka[kom[0], kom[1]] = -1.0
            self._odrysuj_siatke()

    def _stop_rysowania(self, event):
        self.rysowanie = False


    def _zapisz_wzorzec(self):
        if len(self.wzorce) >= MAKS_WZORCOW:
            self._ustaw_status(f"Maksimum {MAKS_WZORCOW} wzorce!", KOL_CZERWONY)
            return
        wzorzec = self.siatka.flatten().copy()
        if np.all(wzorzec == -1.0):
            self._ustaw_status("Siatka jest pusta - nie można zapisać!", KOL_CZERWONY)
            return
        self.wzorce.append(wzorzec)
        self.siatka = np.full((ROZMIAR_SIATKI, ROZMIAR_SIATKI), -1.0)
        self._odrysuj_siatke()
        self._ustaw_status(
            f"Wzorzec {len(self.wzorce)} zapisany. Siatka wyczyszczona.", KOL_ZIELONY
        )
        self._aktualizuj_licznik()
        self._aktualizuj_miniatury()
        self.siec.nauczona = False
        self._aktualizuj_etk_siec()

    def _trenuj(self):
        if not self.wzorce:
            self._ustaw_status("Brak wzorców do trenowania!", KOL_CZERWONY)
            return
        self.siec.naucz(self.wzorce)
        self._ustaw_status(
            f"Sieć nauczona na {len(self.wzorce)} wzorcach.", KOL_ZIELONY
        )
        self._aktualizuj_etk_siec()

    def _dodaj_szum(self):
        ile = 0
        for i in range(ROZMIAR_SIATKI):
            for j in range(ROZMIAR_SIATKI):
                if random.random() < 0.20:
                    self.siatka[i, j] *= -1
                    ile += 1
        self._odrysuj_siatke()
        self._ustaw_status(f"Szum dodany ({ile} pikseli odwrócono).", KOL_TEKST)

    def _rozpoznaj(self):
        if not self.siec.nauczona:
            self._ustaw_status("Najpierw wytrenuj sieć!", KOL_CZERWONY)
            return

        wejscie = self.siatka.flatten().copy()

        if np.all(wejscie == -1.0):
            self._ustaw_status(
                "Panie Hermanowiczu proszę coś wprowadzić", KOL_CZERWONY
            )
            return

        x = wejscie.astype(float)

        for _ in range(40):
            stary_x = x.copy()

            kolejnosc = np.random.permutation(len(x))

            for i in kolejnosc:

                s = np.dot(self.siec.W[i], x) + (random.random() * 0.02 - 0.01)

                nowy = 1.0 if s >= 0 else -1.0

                x[i] = nowy


                self.siatka = x.reshape((ROZMIAR_SIATKI, ROZMIAR_SIATKI))
                self._odrysuj_siatke()

                self.root.update()


                time.sleep(0.03)

            if np.array_equal(x, stary_x):
                break

        self.siatka = x.reshape((ROZMIAR_SIATKI, ROZMIAR_SIATKI))
        self._odrysuj_siatke()

        self._ustaw_status("Rozpoznawanie zakończone.", KOL_ZIELONY)

    def _wyczysc(self):
        self.siatka = np.full((ROZMIAR_SIATKI, ROZMIAR_SIATKI), -1.0)
        self._odrysuj_siatke()
        self._ustaw_status("Siatka wyczyszczona.", KOL_MUTNY)

    def _resetuj_wzorce(self):

        potwierdzenie = messagebox.askyesno(
            "Potwierdzenie",
            "Czy na pewno chcesz usunąć wszystkie wzorce?"
        )

        if not potwierdzenie:
            return

        self.wzorce.clear()
        self.siec.nauczona = False

        self._aktualizuj_licznik()
        self._aktualizuj_miniatury()
        self._aktualizuj_etk_siec()

        self._ustaw_status(
            "Wszystkie wzorce usunięte.",
            KOL_CZERWONY
        )

    def _zaladuj_wzorce_domyslne(self):
        for wzorzec in [WZORZEC_P, WZORZEC_R, WZORZEC_O]:
            self.wzorce.append(wzorzec.flatten().copy())
        self._aktualizuj_licznik()
        self._aktualizuj_miniatury()
        self._ustaw_status(
            "Załadowano wzorce: P, R, O. Kliknij 'Trenuj sieć'.", KOL_AKCENT
        )
    
    def _resize(self, event):


        nowa_szer = self.kanwa.winfo_width()
        nowa_wys = self.kanwa.winfo_height()

        rozmiar = min(nowa_szer, nowa_wys)

        global ROZMIAR_KOMORKI
        ROZMIAR_KOMORKI = max(10, rozmiar // ROZMIAR_SIATKI)

        for i in range(ROZMIAR_SIATKI):
            for j in range(ROZMIAR_SIATKI):

                x1 = j * ROZMIAR_KOMORKI + 1
                y1 = i * ROZMIAR_KOMORKI + 1

                x2 = x1 + ROZMIAR_KOMORKI - 2
                y2 = y1 + ROZMIAR_KOMORKI - 2

                self.kanwa.coords(
                    self.komorki[i][j],
                    x1, y1, x2, y2
                )

        self._odrysuj_siatke()



    def _odrysuj_siatke(self):
        for i in range(ROZMIAR_SIATKI):
            for j in range(ROZMIAR_SIATKI):
                
                kol = KOL_AKTYWNA if self.siatka[i, j] == 1.0 else KOL_KOMORKA
                self.kanwa.itemconfig(self.komorki[i][j], fill=kol)


    def _ustaw_status(self, tekst, kolor=KOL_TEKST):
        self.etk_status.config(text=tekst, fg=kolor)

    def _aktualizuj_licznik(self):
        self.etk_wzorce.config(text=f"Wzorce: {len(self.wzorce)} / {MAKS_WZORCOW}")

    def _aktualizuj_etk_siec(self):
        if self.siec.nauczona:
            self.etk_siec.config(text="Sieć: nauczona ✓", fg=KOL_ZIELONY)
        else:
            self.etk_siec.config(text="Sieć: nienauczona", fg=KOL_CZERWONY)

    def _aktualizuj_miniatury(self):
        for widget in self.ramka_miniatur.winfo_children():
            widget.destroy()

        for idx_w, wzorzec in enumerate(self.wzorce):
            ramka = tk.Frame(self.ramka_miniatur, bg=KOL_PANEL)
            ramka.pack(anchor="w", pady=3)

            tk.Label(
                ramka,
                text=f"#{idx_w + 1}",
                font=("Courier", 9),
                bg=KOL_PANEL,
                fg=KOL_MUTNY,
            ).pack(side=tk.LEFT, padx=(0, 6))

            rozmiar_mini = 4
            wym_mini = ROZMIAR_SIATKI * rozmiar_mini
            mini = tk.Canvas(
                ramka,
                width=wym_mini,
                height=wym_mini,
                bg=KOL_TLO,
                highlightthickness=1,
                highlightbackground=KOL_LINIA,
            )
            mini.pack(side=tk.LEFT)

            mat = wzorzec.reshape((ROZMIAR_SIATKI, ROZMIAR_SIATKI))
            for i in range(ROZMIAR_SIATKI):
                for j in range(ROZMIAR_SIATKI):
                    kol = KOL_AKTYWNA if mat[i, j] == 1.0 else KOL_KOMORKA
                    mini.create_rectangle(
                        j * rozmiar_mini,
                        i * rozmiar_mini,
                        j * rozmiar_mini + rozmiar_mini,
                        i * rozmiar_mini + rozmiar_mini,
                        fill=kol,
                        outline="",
                    )


if __name__ == "__main__":
    root = tk.Tk()
    app = AplikacjaHopfield(root)
    root.mainloop()
