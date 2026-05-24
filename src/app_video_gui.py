from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from video_inference import process_video


class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Body Type Detector")
        self.root.geometry("760x300")
        self.root.minsize(760, 300)

        self.selected_file = tk.StringVar()
        self.status_text = tk.StringVar(value="Status: gotowy")
        self.processing_thread = None
        self.stop_requested = False

        self.build_ui()

    def build_ui(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        title = tk.Label(
            main_frame,
            text="Wykrywanie pojazdów i klasyfikacja typu nadwozia",
            font=("Arial", 15, "bold")
        )
        title.pack(pady=(0, 18))

        file_frame = tk.Frame(main_frame)
        file_frame.pack(fill="x", pady=(0, 16))

        entry = tk.Entry(file_frame, textvariable=self.selected_file, width=80)
        entry.pack(side="left", padx=(0, 10), fill="x", expand=True)

        browse_btn = tk.Button(
            file_frame,
            text="Wybierz plik .mp4",
            width=18,
            command=self.choose_file
        )
        browse_btn.pack(side="left")

        buttons_frame = tk.Frame(main_frame)
        buttons_frame.pack(pady=(0, 18))

        start_btn = tk.Button(
            buttons_frame,
            text="Uruchom analizę",
            width=18,
            command=self.start_processing
        )
        start_btn.pack(side="left", padx=10)

        stop_btn = tk.Button(
            buttons_frame,
            text="Stop",
            width=18,
            command=self.stop_processing
        )
        stop_btn.pack(side="left", padx=10)

        status_label = tk.Label(
            main_frame,
            textvariable=self.status_text,
            font=("Arial", 11)
        )
        status_label.pack(pady=(0, 14))

        info_label = tk.Label(
            main_frame,
            text="Podgląd analizy pojawi się w osobnym oknie OpenCV. Naciśnij q w oknie wideo lub kliknij Stop, aby zakończyć analizę.",
            font=("Arial", 10),
            wraplength=700,
            justify="center"
        )
        info_label.pack(fill="x")

    def choose_file(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz plik wideo",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("All video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.selected_file.set(file_path)
            self.status_text.set("Status: wybrano plik")

    def start_processing(self):
        if self.processing_thread is not None and self.processing_thread.is_alive():
            messagebox.showinfo("Informacja", "Analiza już trwa.")
            return

        file_path = self.selected_file.get().strip()
        if not file_path:
            messagebox.showwarning("Brak pliku", "Najpierw wybierz plik .mp4.")
            return

        path_obj = Path(file_path)
        if not path_obj.exists():
            messagebox.showerror("Błąd", "Wybrany plik nie istnieje.")
            return

        self.stop_requested = False
        self.status_text.set("Status: trwa analiza...")

        self.processing_thread = threading.Thread(
            target=self.run_video_processing,
            args=(path_obj,),
            daemon=True
        )
        self.processing_thread.start()

    def run_video_processing(self, path_obj):
        try:
            process_video(path_obj, stop_flag=lambda: self.stop_requested, frame_step=4)
            self.root.after(0, lambda: self.status_text.set("Status: analiza zakończona"))
        except Exception as e:
            self.root.after(0, lambda: self.status_text.set("Status: błąd"))
            self.root.after(0, lambda: messagebox.showerror("Błąd przetwarzania", str(e)))

    def stop_processing(self):
        self.stop_requested = True
        self.status_text.set("Status: zatrzymywanie...")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()