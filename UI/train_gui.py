import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import json
import os
import sys
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
train_dir = os.path.join(project_root, "train")

# Aggiungi al path
if train_dir not in sys.path:
    sys.path.insert(0, train_dir)

# Import delle funzioni dal train.py
try:
    from train import main as train_main, MODEL_CONFIGS
except ImportError as e:
    print(f"Errore nell'importazione dei moduli: {e}")
    print("Assicurati che train.py sia nella stessa directory e che tutti i pacchetti siano installati.")
    sys.exit(1)

class StatusBar(ttk.Frame):
    """Status bar con messaggi temporanei colorati"""
    def __init__(self, master):
        super().__init__(master, style="StatusBar.TFrame", height=30)
        self.pack_propagate(False)
        self.label = ttk.Label(
            self,
            text="Pronto",
            style="StatusBar.TLabel",
            anchor=tk.W,
            padding=(10, 5)
        )
        self.label.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.timer = None

    def set_message(self, message, msg_type="info", duration=3000):
        """
        Imposta un messaggio temporaneo nella status bar
        msg_type: 'info', 'success', 'warning', 'error'
        duration: millisecondi prima che il messaggio scompaia (0 = permanente)
        """
        if self.timer:
            self.after_cancel(self.timer)
        
        colors = {
            "info": ("#3498db", "white"),
            "success": ("#27ae60", "white"),
            "warning": ("#f39c12", "white"),
            "error": ("#e74c3c", "white")
        }
        
        bg_color, fg_color = colors.get(msg_type, colors["info"])
        style = ttk.Style()
        style.configure("StatusBar.TFrame", background=bg_color)
        style.configure("StatusBar.TLabel", background=bg_color, foreground=fg_color, font=("Segoe UI", 9))
        
        self.label.config(text=message)
        
        if duration > 0:
            self.timer = self.after(duration, self.reset)

    def reset(self):
        """Ripristina la status bar allo stato normale"""
        style = ttk.Style()
        style.configure("StatusBar.TFrame", background="#ecf0f1")
        style.configure("StatusBar.TLabel", background="#ecf0f1", foreground="#2c3e50", font=("Segoe UI", 9))
        self.label.config(text="Pronto")
        self.timer = None

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Trainer - Speech Analysis")
        self.root.geometry("900x750")
        self.root.minsize(850, 600)
        
        # Centra la finestra sullo schermo
        self.center_window()
        self.root.resizable(True, True)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Variabili di configurazione
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # sali di una cartella
        common_dir = os.path.join(project_root, "common")

        # ✅ Crea la cartella common se non esiste (silenzioso)
        os.makedirs(common_dir, exist_ok=True)

        self.config_file = os.path.join(common_dir, "training_config.json")
        self.load_config()
        
        # Variabile per modalità dataset
        self.use_merged_dataset = tk.BooleanVar(value=self.config.get("use_merged_dataset", False))
        
        # Variabili per i path (modalità base)
        self.dataset_custom = tk.StringVar(value=self.config.get("dataset_custom", ""))
        self.dataset_egemaps = tk.StringVar(value=self.config.get("dataset_egemaps", ""))
        self.dataset_index = tk.StringVar(value=self.config.get("dataset_index", ""))
        
        # Variabile per dataset merged
        self.merged_dataset = tk.StringVar(value=self.config.get("merged_dataset", ""))
        self.target_column = tk.StringVar(value=self.config.get("target_column", "Tipo soggetto"))
        
        # Variabili per output paths
        self.mc_summary_csv = tk.StringVar(value=self.config.get("mc_summary_csv", r"risultati\mc_cv_summary.csv"))
        self.model_path = tk.StringVar(value=self.config.get("model_path", r"models\trained_model.pkl"))
        self.test_set_path = tk.StringVar(value=self.config.get("test_set_path", r"models\test_set.csv"))
        self.test_target_path = tk.StringVar(value=self.config.get("test_target_path", r"models\test_target.csv"))
        self.x_columns_path = tk.StringVar(value=self.config.get("x_columns_path", r"models\X_columns.json"))
        
        # Variabile per selezione modello
        self.selected_model = tk.StringVar(value=self.config.get("selected_model", "random_forest"))
        
        # Variabile per n_estimators
        self.use_default_estimators = tk.BooleanVar(value=True)
        self.custom_estimators = tk.StringVar(value="100, 115, 150, 200")
        
        # Flag per training in corso e thread
        self.training_in_progress = False
        self.training_thread = None
        self.stop_training_flag = False
        
        # Applica stile moderno PRIMA di setup_ui
        self.apply_modern_style()
        
        # Setup UI
        self.setup_ui()
        
        # Aggiorna i parametri predefiniti in base al modello
        self.update_default_estimators()

    def load_config(self):
        """Carica la configurazione dal file JSON"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def save_config(self):
        """Salva la configurazione nel file JSON"""
        self.config = {
            "use_merged_dataset": self.use_merged_dataset.get(),
            "dataset_custom": self.dataset_custom.get(),
            "dataset_egemaps": self.dataset_egemaps.get(),
            "dataset_index": self.dataset_index.get(),
            "merged_dataset": self.merged_dataset.get(),
            "target_column": self.target_column.get(),
            "mc_summary_csv": self.mc_summary_csv.get(),
            "model_path": self.model_path.get(),
            "test_set_path": self.test_set_path.get(),
            "test_target_path": self.test_target_path.get(),
            "x_columns_path": self.x_columns_path.get(),
            "selected_model": self.selected_model.get(),
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def center_window(self):
        """Centra la finestra sullo schermo"""
        self.root.update_idletasks()
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2) - int(screen_height * 0.046)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def apply_modern_style(self):
        """Applica uno stile moderno all'applicazione"""
        style = ttk.Style()
        bg_color = "#f5f5f5"
        fg_color = "#2c3e50"
        
        self.root.configure(bg=bg_color)
        
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=fg_color, background=bg_color)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=fg_color, background=bg_color)
        style.configure("TLabel", font=("Segoe UI", 10), foreground=fg_color, background=bg_color)
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=10)
        style.configure("Stop.TButton", font=("Segoe UI", 10), padding=8)
        style.configure("TCheckbutton", font=("Segoe UI", 10), foreground=fg_color, background=bg_color)
        style.configure("TRadiobutton", font=("Segoe UI", 10), foreground=fg_color, background=bg_color)
        style.configure("TFrame", background=bg_color)
        style.configure("Card.TFrame", background="white", relief="flat")
        style.configure("StatusBar.TFrame", background="#ecf0f1")
        style.configure("StatusBar.TLabel", background="#ecf0f1", foreground=fg_color, font=("Segoe UI", 9))

    def setup_ui(self):
        """Setup dell'interfaccia utente"""
        # Status Bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Container principale
        container = ttk.Frame(self.root, style="TFrame")
        container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(container, style="TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        title_label = ttk.Label(header_frame, text="🎤 ML Speech Classifier", style="Title.TLabel")
        title_label.pack(side=tk.LEFT)
        
        settings_btn = ttk.Button(header_frame, text="⚙️ Impostazioni", command=self.open_settings)
        settings_btn.pack(side=tk.RIGHT)
        
        # Separator
        ttk.Separator(container, orient='horizontal').pack(fill=tk.X, padx=20, pady=5)
        
        # Main content frame
        main_frame = ttk.Frame(container, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # LEFT PANEL
        left_panel = ttk.Frame(main_frame, style="Card.TFrame")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        config_label = ttk.Label(left_panel, text="Configurazione Training", style="Header.TLabel")
        config_label.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        content_wrapper = ttk.Frame(left_panel, style="Card.TFrame")
        content_wrapper.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # AREA SCROLLABILE
        scroll_canvas = tk.Canvas(content_wrapper, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_wrapper, orient="vertical", command=scroll_canvas.yview)
        
        scrollable_config = ttk.Frame(scroll_canvas, style="Card.TFrame")
        scrollable_config.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        
        canvas_window = scroll_canvas.create_window((0, 0), window=scrollable_config, anchor="nw")
        
        def on_canvas_configure(event):
            scroll_canvas.itemconfig(canvas_window, width=event.width)
        
        scroll_canvas.bind("<Configure>", on_canvas_configure)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            scroll_canvas.unbind_all("<MouseWheel>")
        
        scroll_canvas.bind('<Enter>', _bind_to_mousewheel)
        scroll_canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === MODALITÀ DATASET BASE ===
        mode_frame = ttk.LabelFrame(scrollable_config, text="Modalità Dataset", padding=15)
        mode_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        traditional_radio = ttk.Radiobutton(
            mode_frame,
            text="Dataset custom-egemaps-index (merging 3 file)",
            variable=self.use_merged_dataset,
            value=False,
            command=self.toggle_dataset_mode
        )
        traditional_radio.pack(anchor=tk.W, pady=(0, 5))
        
        merged_radio = ttk.Radiobutton(
            mode_frame,
            text="Dataset già merged (feature + target)",
            variable=self.use_merged_dataset,
            value=True,
            command=self.toggle_dataset_mode
        )
        merged_radio.pack(anchor=tk.W)
        
        # Selezione Modello
        model_frame = ttk.LabelFrame(scrollable_config, text="Selezione Modello", padding=15)
        model_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        ttk.Label(model_frame, text="Tipo di modello:", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(0, 5))
        
        model_keys = list(MODEL_CONFIGS.keys())
        model_names = [MODEL_CONFIGS[key]['display_name'] for key in model_keys]
        
        self.model_combo = ttk.Combobox(
            model_frame,
            values=model_names,
            state="readonly",
            width=35
        )
        
        initial_idx = model_keys.index(self.selected_model.get()) if self.selected_model.get() in model_keys else 0
        self.model_combo.current(initial_idx)
        self.model_combo.pack(fill=tk.X, pady=(0, 5))
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self.on_model_change())
        
        self.model_info_label = ttk.Label(
            model_frame,
            text="",
            font=("Segoe UI", 9, "italic"),
            foreground="#7f8c8d"
        )
        self.model_info_label.pack(anchor=tk.W)
        
        # Parametri Monte Carlo
        estimators_frame = ttk.LabelFrame(scrollable_config, text="Parametri Monte Carlo", padding=15)
        estimators_frame.pack(fill=tk.X, padx=15, pady=10)
        
        default_radio = ttk.Radiobutton(
            estimators_frame,
            text="Usa valori predefiniti",
            variable=self.use_default_estimators,
            value=True,
            command=self.toggle_estimators_entry
        )
        default_radio.pack(anchor=tk.W, pady=(0, 5))
        
        self.default_label = ttk.Label(
            estimators_frame,
            text="",
            font=("Segoe UI", 9, "italic"),
            foreground="#7f8c8d"
        )
        self.default_label.pack(anchor=tk.W, padx=(20, 0), pady=(0, 8))
        
        custom_radio = ttk.Radiobutton(
            estimators_frame,
            text="Personalizza parametri:",
            variable=self.use_default_estimators,
            value=False,
            command=self.toggle_estimators_entry
        )
        custom_radio.pack(anchor=tk.W, pady=(0, 5))
        
        self.estimators_entry = ttk.Entry(estimators_frame, textvariable=self.custom_estimators, width=40)
        self.estimators_entry.pack(fill=tk.X, pady=(0, 5))
        self.estimators_entry.config(state='disabled')
        
        hint_label = ttk.Label(estimators_frame, text="Formato: numeri separati da virgola (es: 50, 100, 150)",
                              font=("Segoe UI", 9, "italic"), foreground="#7f8c8d")
        hint_label.pack(anchor=tk.W)
        
        # Dataset status
        self.status_frame = ttk.LabelFrame(scrollable_config, text="Stato Dataset", padding=15)
        self.status_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.status_labels = {}
        self.setup_dataset_status()
        
        # AREA FISSA IN FONDO
        fixed_bottom_frame = ttk.Frame(left_panel, style="Card.TFrame")
        fixed_bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(10, 15))
        
        self.train_button = ttk.Button(
            fixed_bottom_frame,
            text="▶ Avvia Training",
            style="Accent.TButton",
            command=self.start_training
        )
        self.train_button.pack(fill=tk.X, ipady=10)
        
        self.progress = ttk.Progressbar(fixed_bottom_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        # RIGHT PANEL
        right_panel = ttk.Frame(main_frame, style="Card.TFrame")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        output_label = ttk.Label(right_panel, text="Output Training", style="Header.TLabel")
        output_label.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        self.output_text = scrolledtext.ScrolledText(
            right_panel,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            relief=tk.FLAT
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        output_buttons_frame = ttk.Frame(right_panel)
        output_buttons_frame.pack(anchor=tk.E, padx=15, pady=(0, 15))
        
        self.stop_btn = ttk.Button(
            output_buttons_frame,
            text="⏹ Stop Training",
            command=self.stop_training,
            style="Stop.TButton"
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn.pack_forget()
        
        clear_btn = ttk.Button(output_buttons_frame, text="🗑 Pulisci Output", command=self.clear_output)
        clear_btn.pack(side=tk.LEFT)

    def setup_dataset_status(self):
        """Setup dello status dei dataset in base alla modalità"""
        # Rimuovi tutti i widget esistenti
        for widget in self.status_frame.winfo_children():
            widget.destroy()
        
        self.status_labels.clear()
        
        if self.use_merged_dataset.get():
            # Modalità merged
            datasets = [
                ("Merged Dataset:", self.merged_dataset),
            ]
        else:
            # Modalità base
            datasets = [
                ("Custom Dataset:", self.dataset_custom),
                ("eGeMAPS Features:", self.dataset_egemaps),
                ("Dataset Index:", self.dataset_index)
            ]
        
        for label_text, var in datasets:
            row_frame = ttk.Frame(self.status_frame)
            row_frame.pack(fill=tk.X, pady=3)
            
            ttk.Label(row_frame, text=label_text, width=18).pack(side=tk.LEFT)
            
            path = var.get()
            status = "✓" if path and os.path.exists(path) else "✗"
            color = "#27ae60" if path and os.path.exists(path) else "#e74c3c"
            
            status_label = ttk.Label(row_frame, text=status, foreground=color, font=("Segoe UI", 12, "bold"))
            status_label.pack(side=tk.LEFT)
            
            self.status_labels[label_text] = status_label

    def toggle_dataset_mode(self):
        """Cambia modalità dataset"""
        self.setup_dataset_status()
        self.save_config()

    def on_model_change(self):
        """Chiamato quando l'utente cambia modello dal dropdown"""
        model_keys = list(MODEL_CONFIGS.keys())
        selected_idx = self.model_combo.current()
        model_key = model_keys[selected_idx]
        self.selected_model.set(model_key)
        self.update_default_estimators()

    def update_default_estimators(self):
        """Aggiorna i valori predefiniti in base al modello selezionato"""
        model_type = self.selected_model.get()
        if model_type in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_type]
            param_name = config['param_name']
            default_grid = config['default_grid']
            
            self.model_info_label.config(
                text=f"Parametro ottimizzato: {param_name}"
            )
            
            if param_name == 'hidden_layer_sizes':
                default_str = f"({', '.join(map(str, default_grid))})"
            else:
                default_str = f"({', '.join(map(str, default_grid))})"
            
            self.default_label.config(text=default_str)
            
            if param_name == 'hidden_layer_sizes':
                formatted = ', '.join([str(x) if isinstance(x, tuple) else f"({x},)" for x in default_grid])
                self.custom_estimators.set(formatted)
            else:
                self.custom_estimators.set(', '.join(map(str, default_grid)))

    def toggle_estimators_entry(self):
        """Abilita/disabilita l'entry per parametri personalizzati"""
        if self.use_default_estimators.get():
            self.estimators_entry.config(state='disabled')
        else:
            self.estimators_entry.config(state='normal')

    def open_settings(self):
        """Apre la finestra delle impostazioni"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Impostazioni")
        settings_window.geometry("700x650")
        settings_window.minsize(600, 500)
        settings_window.transient(self.root)
        settings_window.grab_set()
        settings_window.configure(bg="#f5f5f5")
        
        container = ttk.Frame(settings_window, padding=20)
        container.pack(fill=tk.BOTH, expand=True)
        
        title = ttk.Label(container, text="Configurazione Percorsi", style="Header.TLabel")
        title.pack(anchor=tk.W, pady=(0, 20))
        
        scroll_container = ttk.Frame(container)
        scroll_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        canvas = tk.Canvas(scroll_container, highlightthickness=0, bg="#f5f5f5")
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Dataset base (3 componenti)
        input_frame = ttk.LabelFrame(scrollable_frame, text="Dataset Input (Modalità base)", padding=15)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.create_path_selector(input_frame, "Custom Dataset:", self.dataset_custom, "csv")
        self.create_path_selector(input_frame, "eGeMAPS Features:", self.dataset_egemaps, "csv")
        self.create_path_selector(input_frame, "Dataset Index:", self.dataset_index, "xlsx")
        
        # Dataset merged
        merged_frame = ttk.LabelFrame(scrollable_frame, text="Dataset Merged (Modalità Merged)", padding=15)
        merged_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.create_path_selector(merged_frame, "Merged Dataset:", self.merged_dataset, "csv")
        
        # Target column
        target_frame = ttk.Frame(merged_frame)
        target_frame.pack(fill=tk.X, pady=5)
        ttk.Label(target_frame, text="Target Column:", width=18).pack(side=tk.LEFT, padx=(0, 10))
        target_entry = ttk.Entry(target_frame, textvariable=self.target_column)
        target_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Output paths
        output_frame = ttk.LabelFrame(scrollable_frame, text="Percorsi Output", padding=15)
        output_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.create_path_selector(output_frame, "MC Summary CSV:", self.mc_summary_csv, "csv", is_output=True)
        self.create_path_selector(output_frame, "Model Path:", self.model_path, "pkl", is_output=True)
        self.create_path_selector(output_frame, "Test Set:", self.test_set_path, "csv", is_output=True)
        self.create_path_selector(output_frame, "Test Target:", self.test_target_path, "csv", is_output=True)
        self.create_path_selector(output_frame, "X Columns JSON:", self.x_columns_path, "json", is_output=True)
        
        separator = ttk.Separator(container, orient='horizontal')
        separator.pack(fill=tk.X, pady=(0, 15))
        
        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X)
        
        style = ttk.Style()
        style.configure("SmallAccent.TButton", padding=(10, 6))
        
        save_btn = ttk.Button(
            button_frame,
            text="💾 Salva e Chiudi",
            command=lambda: self.save_settings(settings_window),
            style="SmallAccent.TButton"
        )
        save_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        def on_close():
            canvas.unbind('<Enter>')
            canvas.unbind('<Leave>')
            canvas.unbind_all("<MouseWheel>")
            settings_window.destroy()
        
        settings_window.protocol("WM_DELETE_WINDOW", on_close)

    def create_path_selector(self, parent, label_text, variable, file_type, is_output=False):
        """Crea un selettore di percorso file"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        label = ttk.Label(frame, text=label_text, width=18)
        label.pack(side=tk.LEFT, padx=(0, 10))
        
        entry = ttk.Entry(frame, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(
            frame,
            text="Sfoglia",
            width=10,
            command=lambda: self.browse_file(variable, file_type, is_output)
        )
        browse_btn.pack(side=tk.RIGHT)

    def browse_file(self, variable, file_type, is_output=False):
        """Apre il dialog per selezionare un file"""
        file_types = {
            "csv": [("CSV files", "*.csv"), ("All files", "*.*")],
            "xlsx": [("Excel files", "*.xlsx"), ("All files", "*.*")],
            "pkl": [("Pickle files", "*.pkl"), ("All files", "*.*")],
            "json": [("JSON files", "*.json"), ("All files", "*.*")]
        }
        
        if is_output:
            filename = filedialog.asksaveasfilename(
                title="Seleziona percorso di salvataggio",
                filetypes=file_types.get(file_type, [("All files", "*.*")])
            )
        else:
            filename = filedialog.askopenfilename(
                title="Seleziona file",
                filetypes=file_types.get(file_type, [("All files", "*.*")])
            )
        
        if filename:
            variable.set(filename)

    def save_settings(self, window):
        """Salva le impostazioni e chiude la finestra"""
        self.save_config()
        self.update_status_labels()
        self.status_bar.set_message("✓ Impostazioni salvate correttamente", "success", 3000)
        window.destroy()

    def update_status_labels(self):
        """Aggiorna le etichette di stato dei dataset"""
        if self.use_merged_dataset.get():
            datasets = [
                ("Merged Dataset:", self.merged_dataset),
            ]
        else:
            datasets = [
                ("Custom Dataset:", self.dataset_custom),
                ("eGeMAPS Features:", self.dataset_egemaps),
                ("Dataset Index:", self.dataset_index)
            ]
        
        for label_text, var in datasets:
            path = var.get()
            exists = path and os.path.exists(path)
            status = "✓" if exists else "✗"
            color = "#27ae60" if exists else "#e74c3c"
            
            if label_text in self.status_labels:
                self.status_labels[label_text].config(text=status, foreground=color)
        
        self.root.update_idletasks()

    def clear_output(self):
        """Pulisce l'output della console"""
        self.output_text.delete(1.0, tk.END)
        if not self.training_in_progress:
            self.status_bar.set_message("Output pulito", "info", 2000)

    def log_output(self, message):
        """Aggiunge un messaggio all'output"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()

    def validate_inputs(self):
        """Valida che tutti i percorsi necessari siano impostati"""
        if self.use_merged_dataset.get():
            # Modalità merged
            if not self.merged_dataset.get() or not os.path.exists(self.merged_dataset.get()):
                self.status_bar.set_message("✗ Errore: Merged Dataset non trovato", "error", 5000)
                messagebox.showerror("Errore", "Merged Dataset non trovato o non impostato.\nVerifica le impostazioni.")
                return False
        else:
            # Modalità base
            required_paths = [
                ("Custom Dataset", self.dataset_custom.get()),
                ("eGeMAPS Features", self.dataset_egemaps.get()),
                ("Dataset Index", self.dataset_index.get())
            ]
            
            for name, path in required_paths:
                if not path or not os.path.exists(path):
                    self.status_bar.set_message(f"✗ Errore: {name} non trovato", "error", 5000)
                    messagebox.showerror("Errore", f"{name} non trovato o non impostato.\nVerifica le impostazioni.")
                    return False
        
        return True

    def parse_estimators(self):
        """Parsea la stringa di parametri in base al modello selezionato"""
        if self.use_default_estimators.get():
            return None
        
        model_type = self.selected_model.get()
        config = MODEL_CONFIGS[model_type]
        param_name = config['param_name']
        
        try:
            estimators_str = self.custom_estimators.get()
            
            if param_name in ['n_estimators']:
                estimators = [int(x.strip()) for x in estimators_str.split(',')]
            elif param_name in ['C', 'learning_rate', 'alpha']:
                estimators = [float(x.strip()) for x in estimators_str.split(',')]
            elif param_name == 'hidden_layer_sizes':
                estimators = []
                for x in estimators_str.split(','):
                    x = x.strip()
                    if '(' in x:
                        nums = x.strip('()').split(',')
                        estimators.append(tuple(int(n.strip()) for n in nums))
                    else:
                        estimators.append((int(x),))
            else:
                estimators = [x.strip() for x in estimators_str.split(',')]
            
            if not estimators:
                raise ValueError("Lista vuota")
            
            return estimators
        
        except ValueError:
            self.status_bar.set_message(f"✗ Formato parametri non valido", "error", 5000)
            messagebox.showerror("Errore", f"Formato parametri non valido per {param_name}.\nUsa il formato corretto (es: 50, 100, 150)")
            return None

    def start_training(self):
        """Avvia il processo di training in un thread separato"""
        if self.training_in_progress:
            self.status_bar.set_message("⚠ Training già in esecuzione", "warning", 3000)
            return
        
        if not self.validate_inputs():
            return
        
        estimators = self.parse_estimators()
        if estimators is None and not self.use_default_estimators.get():
            return
        
        model_type = self.selected_model.get()
        model_name = MODEL_CONFIGS[model_type]['display_name']
        
        self.training_in_progress = True
        self.stop_training_flag = False
        self.train_button.config(state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.progress.start(10)
        
        self.clear_output()
        self.log_output(f"=== Inizio Training: {model_name} ===\n")
        
        if self.use_merged_dataset.get():
            self.log_output(f"Modalità: Dataset Merged\n")
        else:
            self.log_output(f"Modalità: Dataset base 3 componenti\n")
        
        if estimators:
            param_name = MODEL_CONFIGS[model_type]['param_name']
            self.log_output(f"{param_name} da testare: {estimators}\n")
        else:
            self.log_output(f"Utilizzo parametri predefiniti\n")
        
        self.status_bar.set_message(f"▶ Training {model_name} in corso...", "info", 0)
        
        self.training_thread = threading.Thread(target=self.run_training, args=(estimators, model_type))
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_training(self):
        """Ferma il training in corso"""
        if self.training_in_progress:
            self.stop_training_flag = True
            self.log_output("\n⚠️ Interruzione training richiesta...")
            self.status_bar.set_message("⏳ Interruzione in corso...", "warning", 0)

    def run_training(self, estimators, model_type):
        """Esegue il training (chiamato in un thread separato)"""
        try:
            original_stdout = sys.stdout
            
            class OutputRedirector:
                def __init__(self, callback, stop_flag_callback):
                    self.callback = callback
                    self.stop_flag_callback = stop_flag_callback
                
                def write(self, message):
                    if self.stop_flag_callback():
                        raise KeyboardInterrupt("Training interrotto dall'utente")
                    if message.strip():
                        self.callback(message)
                
                def flush(self):
                    pass
            
            sys.stdout = OutputRedirector(self.log_output, lambda: self.stop_training_flag)
            
            # Prepara gli argomenti in base alla modalità
            if self.use_merged_dataset.get():
                train_main(
                    dataset_custom_path=None,
                    dataset_egemaps_path=None,
                    dataset_index_path=None,
                    mc_summary_csv=self.mc_summary_csv.get(),
                    model_path=self.model_path.get(),
                    test_set_path=self.test_set_path.get(),
                    test_target_path=self.test_target_path.get(),
                    x_columns_path=self.x_columns_path.get(),
                    grid_param_list=estimators,
                    model_type=model_type,
                    merged_dataset_path=self.merged_dataset.get(),
                    target_column=self.target_column.get()
                )
            else:
                train_main(
                    dataset_custom_path=self.dataset_custom.get(),
                    dataset_egemaps_path=self.dataset_egemaps.get(),
                    dataset_index_path=self.dataset_index.get(),
                    mc_summary_csv=self.mc_summary_csv.get(),
                    model_path=self.model_path.get(),
                    test_set_path=self.test_set_path.get(),
                    test_target_path=self.test_target_path.get(),
                    x_columns_path=self.x_columns_path.get(),
                    grid_param_list=estimators,
                    model_type=model_type,
                    merged_dataset_path=None,
                    target_column=self.target_column.get()
                )
            
            sys.stdout = original_stdout
            
            if self.stop_training_flag:
                self.root.after(0, self.training_complete, False, "Training interrotto dall'utente")
            else:
                self.root.after(0, self.training_complete, True, None)
        
        except KeyboardInterrupt:
            sys.stdout = original_stdout
            self.root.after(0, self.training_complete, False, "Training interrotto dall'utente")
        
        except Exception as e:
            sys.stdout = original_stdout
            self.root.after(0, self.training_complete, False, str(e))

    def training_complete(self, success, error_msg):
        """Chiamato quando il training è completato"""
        self.training_in_progress = False
        self.stop_training_flag = False
        self.train_button.config(state='normal')
        self.stop_btn.pack_forget()
        self.progress.stop()
        
        if success:
            self.log_output("\n=== Training Completato con Successo! ===")
            self.status_bar.set_message("✓ Training completato con successo!", "success", 5000)
        else:
            if "interrotto" in error_msg.lower():
                self.log_output(f"\n🔴 === TRAINING INTERROTTO === 🔴")
                self.status_bar.set_message("⏹ Training interrotto dall'utente", "warning", 4000)
                self.root.after(1000, self.clear_output)
            else:
                self.log_output(f"\n=== Errore durante il Training ===\n{error_msg}")
                self.status_bar.set_message(f"✗ Errore: {error_msg[:50]}...", "error", 6000)
                messagebox.showerror("Errore", f"Si è verificato un errore durante il training:\n{error_msg}")

def main():
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
