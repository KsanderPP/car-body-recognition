from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import shutil
import tempfile

from video_inference import process_video


class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Body Type Detector")
        self.root.geometry("860x360")
        self.root.minsize(860, 360)

        self.selected_file = tk.StringVar()
        self.status_text = tk.StringVar(value="Status: gotowy")

        self.processing_thread = None
        self.stop_requested = False

        self.last_stats = None
        self.last_input_path = None
        self.temp_output_video_path = None

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

        input_frame = tk.LabelFrame(main_frame, text="Plik wejściowy", padx=10, pady=10)
        input_frame.pack(fill="x", pady=(0, 14))

        input_entry = tk.Entry(input_frame, textvariable=self.selected_file, width=95)
        input_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)

        browse_input_btn = tk.Button(
            input_frame,
            text="Wybierz plik .mp4",
            width=18,
            command=self.choose_input_file
        )
        browse_input_btn.pack(side="left")

        buttons_frame = tk.Frame(main_frame)
        buttons_frame.pack(pady=(6, 18))

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

        save_video_btn = tk.Button(
            buttons_frame,
            text="Zapisz przetworzony film",
            width=22,
            command=self.save_processed_video
        )
        save_video_btn.pack(side="left", padx=10)

        save_stats_btn = tk.Button(
            buttons_frame,
            text="Zapisz statystyki",
            width=18,
            command=self.save_stats
        )
        save_stats_btn.pack(side="left", padx=10)

        status_label = tk.Label(
            main_frame,
            textvariable=self.status_text,
            font=("Arial", 11)
        )
        status_label.pack(pady=(0, 14))

        info_label = tk.Label(
            main_frame,
            text="Po zakończeniu analizy możesz samodzielnie zapisać przetworzony film i statystyki do wybranego miejsca na komputerze.",
            font=("Arial", 10),
            wraplength=800,
            justify="center"
        )
        info_label.pack(fill="x")

    def choose_input_file(self):
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
            self.last_input_path = Path(file_path)
            self.status_text.set("Status: wybrano plik wejściowy")

    def get_default_result_stem(self):
        if self.last_input_path is not None:
            return f"{self.last_input_path.stem}_result"
        return "video_result"

    def format_stats(self, stats):
        lines = []
        lines.append("Podsumowanie analizy")
        lines.append("")
        lines.append(f"Liczba przetworzonych klatek: {stats['processed_frames']}")
        lines.append(f"Czas analizy [s]: {stats['elapsed_time_sec']:.2f}")
        lines.append(f"Średni FPS przetwarzania: {stats['average_fps']:.2f}")
        lines.append(f"Liczba wykryć pojazdów: {stats['total_detections']}")
        lines.append("")

        lines.append("Wykrycia per typ nadwozia:")
        detections = stats["detections_per_class"]
        if detections:
            for class_name, count in sorted(detections.items()):
                lines.append(f" - {class_name}: {count}")
        else:
            lines.append(" - brak wykryć")

        lines.append("")
        lines.append("Średnia pewność klasyfikacji per klasa:")
        avg_conf = stats["average_confidence_per_class"]

        any_conf = False
        for class_name, value in avg_conf.items():
            if value > 0:
                lines.append(f" - {class_name}: {value:.3f}")
                any_conf = True

        if not any_conf:
            lines.append(" - brak danych")

        lines.append("")
        lines.append("Uwaga: statystyki dotyczą liczby wykryć, a nie liczby unikalnych pojazdów.")

        return "\n".join(lines)

    def save_stats_to_file(self, stats, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() == ".json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.format_stats(stats))

    def start_processing(self):
        if self.processing_thread is not None and self.processing_thread.is_alive():
            messagebox.showinfo("Informacja", "Analiza już trwa.")
            return

        file_path = self.selected_file.get().strip()
        if not file_path:
            messagebox.showwarning("Brak pliku", "Najpierw wybierz plik wejściowy .mp4.")
            return

        path_obj = Path(file_path)
        if not path_obj.exists():
            messagebox.showerror("Błąd", "Wybrany plik wejściowy nie istnieje.")
            return

        self.stop_requested = False
        self.last_stats = None
        self.last_input_path = path_obj

        temp_dir = Path(tempfile.gettempdir())
        self.temp_output_video_path = temp_dir / f"{self.get_default_result_stem()}_temp.mp4"

        self.status_text.set("Status: trwa analiza...")

        self.processing_thread = threading.Thread(
            target=self.run_video_processing,
            args=(path_obj,),
            daemon=True
        )
        self.processing_thread.start()

    def run_video_processing(self, path_obj):
        try:
            stats = process_video(
                path_obj,
                stop_flag=lambda: self.stop_requested,
                frame_step=4,
                output_video_path=self.temp_output_video_path
            )

            self.last_stats = stats
            stats_message = self.format_stats(stats)

            self.root.after(0, lambda: self.status_text.set("Status: analiza zakończona"))
            self.root.after(0, lambda: messagebox.showinfo("Statystyki analizy", stats_message))
        except Exception as e:
            self.root.after(0, lambda: self.status_text.set("Status: błąd"))
            self.root.after(0, lambda: messagebox.showerror("Błąd przetwarzania", str(e)))

    def stop_processing(self):
        self.stop_requested = True
        self.status_text.set("Status: zatrzymywanie...")

    def save_processed_video(self):
        if self.temp_output_video_path is None or not Path(self.temp_output_video_path).exists():
            messagebox.showwarning("Brak filmu", "Nie ma jeszcze przetworzonego filmu do zapisania.")
            return

        default_name = f"{self.get_default_result_stem()}.mp4"

        file_path = filedialog.asksaveasfilename(
            title="Zapisz przetworzony film jako",
            defaultextension=".mp4",
            initialfile=default_name,
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            shutil.copy2(self.temp_output_video_path, file_path)
            self.status_text.set("Status: zapisano przetworzony film")
            messagebox.showinfo("Sukces", f"Zapisano film do:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", str(e))

    def save_stats(self):
        if self.last_stats is None:
            messagebox.showwarning("Brak statystyk", "Nie ma jeszcze statystyk do zapisania.")
            return

        default_name = f"{self.get_default_result_stem()}.json"

        file_path = filedialog.asksaveasfilename(
            title="Zapisz statystyki jako",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            self.save_stats_to_file(self.last_stats, file_path)
            self.status_text.set("Status: zapisano statystyki")
            messagebox.showinfo("Sukces", f"Zapisano statystyki do:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()