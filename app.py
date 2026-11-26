"""Tkinter front end. Evaluation and subprocess I/O stay off the UI thread."""

import json
from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

from environments import BUILTINS
from evolution import Settings, TrainingState, train
from network import TEMPLATES
from runs import DEFAULT_RUNS, Experiment, load_run, save_run


BG = "#11151c"
PANEL = "#1a202a"
FIELD = "#232c38"
EDGE = "#374253"
TEXT = "#f2f5fa"
MUTED = "#a9b6c8"
ACCENT = "#91edc8"
TINT = "#203e38"
BLUE = "#a9bafa"
STAGE = "#141d2d"
STAGE_TEXT = "#eef4fb"
STAGE_MUTED = "#acbcd1"
MODES = {"Melhor agente": "best", "Todos os agentes": "population"}
SPEEDS = {"Normal": 3, "Rápido": 1, "Headless": None}
ANIMATION_TICK_MS = 16
MODE_HELP = {
    "best": "Veja quem conseguiu a melhor pontuação nesta rodada.",
    "population": "Veja todos os agentes tentando resolver o desafio.",
    "none": "O treino avança sem animação. As pontuações continuam aparecendo.",
}
DEMO_ENV = "Desvie dos obstáculos"
FILE_ENV = "Importar desafio"
CHALLENGES = {
    DEMO_ENV: ("navigation", "Desvie dos obstáculos para chegar ao alvo verde."),
    "Voo das lanternas": ("lantern-flight", "Ganhe altitude para passar pelas aberturas na caverna."),
    "Corrida lunar": ("moon-run", "Salte sobre as barreiras e avance pela superfície lunar."),
}
JSON_ARCH = "JSON personalizado"
PYTHON_ARCH = "Arquivo Python"
PRESETS = {
    "Equilibrada": ("Compacta · 16", "Uma boa escolha para começar."),
    "Mais completa": ("Multicamadas · 32 × 16", "Uma rede maior para desafios mais complexos."),
    "Mais simples": ("Linear", "Uma rede menor para testar ideias rapidamente."),
}
FIELDS = [
    ("population", "Agentes por rodada", int), ("generations", "Rodadas de treino", int),
    ("elites", "Agentes preservados", int), ("parents", "Grupo de pais", int),
    ("mutation_rate", "Taxa de mutação", float),
    ("mutation_scale", "Magnitude da mutação", float), ("seed", "Semente", int),
]
FIELD_HELP = {
    "elites": "Indivíduos preservados sem alterações.",
    "parents": "Melhores indivíduos aptos a gerar descendentes.",
    "mutation_rate": "Probabilidade de mutar cada peso, de 0 a 1.",
    "mutation_scale": "Desvio padrão do ruído aplicado aos pesos.",
    "seed": "Use o mesmo valor para repetir o sorteio.",
}


def motion_between(frame, following):
    """Match moving shapes by their stable IDs across two keyframes."""
    sources = {shape["id"]: shape for shape in frame["shapes"]
               if isinstance(shape.get("id"), (str, int))}
    if not sources or following is None:
        return bool(sources), {}
    width, height = frame["width"], frame["height"]
    if (width, height) != (following["width"], following["height"]):
        return True, {}
    motion = {}
    for target in following["shapes"]:
        key = target.get("id")
        if not isinstance(key, (str, int)) or key not in sources:
            continue
        source = sources[key]
        if source["kind"] != target["kind"] or target["kind"] not in ("circle", "rect"):
            continue
        dx, dy = target["x"] - source["x"], target["y"] - source["y"]
        if (dx or dy) and abs(dx) <= width / 2 and abs(dy) <= height / 2:
            motion[key] = (dx, dy)
    return True, motion


class Session:
    def __init__(self, experiment, settings, view, *, saved=None, replay=False):
        self.lock = Lock()
        self.stop = Event()
        self.played = Event()
        self.history = deque(maxlen=180)
        self.latest = None
        self.pending = None
        self.done = False
        self.error = None
        self.saved = None
        self.replay = replay
        self.view = view
        Thread(target=self.run, args=(experiment, settings, saved), daemon=True).start()

    def run(self, experiment, settings, saved):
        state = TrainingState()
        initial = 0
        try:
            if self.replay:
                frames = saved.replay()
                with self.lock:
                    if self.stop.is_set():
                        return
                    self.played.clear()
                    self.pending = frames
                self.played.wait()
                return
            if saved:
                settings, state = saved.continuation(settings)
                initial = state.completed
                with self.lock:
                    self.history.extend(row[:3] for row in state.rows[-self.history.maxlen:])
            with experiment.open() as (environment, network):
                train(environment, network, settings, self.get_view, self.stop, self.publish, state=state)
        except Exception as error:
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"
        finally:
            if not self.replay and state.completed > initial:
                outcome = "failed" if self.error else "interrupted" if self.stop.is_set() else "completed"
                try:
                    self.saved = save_run(experiment, settings, state, outcome)
                except Exception as error:
                    with self.lock:
                        self.error = f"{self.error + ' · ' if self.error else ''}Falha ao salvar o treino. {error}"
            with self.lock:
                self.done = True

    def get_view(self):
        with self.lock:
            return self.view

    def set_view(self, view):
        with self.lock:
            self.view = view
            self.pending = None
            self.played.set()

    def publish(self, generation):
        with self.lock:
            self.latest = (generation.number, generation.best, generation.mean,
                           generation.record, generation.seconds)
            self.history.append(self.latest[:3])
            if generation.view != self.view or not generation.frames or self.stop.is_set():
                return
            self.played.clear()
            self.pending = generation.frames
        self.played.wait()

    def cancel(self):
        with self.lock:
            self.stop.set()
            self.pending = None
            self.played.set()

    def snapshot(self):
        with self.lock:
            frames, self.pending = self.pending, None
            return self.latest, frames, self.done, self.error


class App:
    def __init__(self, root):
        self.root = root
        self.session = None
        self.saved_run = None
        self.latest = None
        self.frames = []
        self.frame_index = 0
        self.frame_job = None
        self.frame_duration_ms = 0
        self.frame_started = 0.0
        self.frame_motion = {}
        self.interpolated_clip = False
        self.motion_items = []
        self.motion_fraction = 0.0
        self.current_frame = None
        self.chart_data = []
        self.chart_geometry = None
        self.custom_json = None
        self.previous_architecture = None
        self.view_message = ("Ambiente pronto", "Inicie o treino para acompanhar a avaliação.")
        self.total = 0
        self.closing = False
        root.title("EvoLab | Treino evolutivo")
        root.geometry("1160x790")
        root.minsize(900, 710)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.style()
        self.layout()
        self.change_environment()
        root.after(30, self.poll)

    def style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Shell.TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL, borderwidth=1, relief="solid",
                        bordercolor=EDGE, lightcolor=EDGE, darkcolor=EDGE)
        style.configure("Title.TLabel", background=BG, font=("Segoe UI", 26, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED)
        style.configure("Heading.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Field.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Metric.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Best.Metric.TLabel", foreground=ACCENT)
        style.configure("Mean.Metric.TLabel", foreground=BLUE)
        style.configure("Status.TLabel", background=FIELD, foreground=MUTED,
                        font=("Segoe UI", 10, "bold"), padding=(12, 6))
        style.configure("Active.Status.TLabel", background=TINT, foreground=ACCENT)
        style.configure("Failed.Status.TLabel", background="#492b32", foreground="#ffb7b3")
        style.configure("Error.TLabel", background="#492b32", foreground="#ffb7b3", padding=(10, 6))
        style.configure("TButton", background=FIELD, bordercolor=EDGE, lightcolor=EDGE,
                        darkcolor=EDGE, focuscolor=ACCENT, padding=(10, 6))
        style.map("TButton", background=[("pressed", EDGE), ("active", "#2c3747")],
                  foreground=[("disabled", MUTED)])
        style.configure("Start.TButton", background=ACCENT, foreground="#10251e",
                        bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                        focuscolor="#10251e", padding=(12, 10), font=("Segoe UI", 11, "bold"))
        style.map("Start.TButton", background=[("active", "#b2f5dc")])
        style.configure("Secondary.TButton", background=PANEL, foreground=MUTED,
                        padding=(8, 7), anchor="w")
        style.configure("Mode.TRadiobutton", indicatoron=False, background=FIELD, bordercolor=EDGE,
                        lightcolor=EDGE, darkcolor=EDGE, focuscolor=ACCENT, padding=(9, 5),
                        font=("Segoe UI", 9, "bold"))
        style.map("Mode.TRadiobutton", background=[("selected", TINT), ("active", "#2c3747")],
                  foreground=[("selected", ACCENT)])
        style.layout("Mode.TRadiobutton", style.layout("TButton"))
        for name in ("TCombobox", "TEntry"):
            style.configure(name, fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                            bordercolor=EDGE, lightcolor=EDGE, darkcolor=EDGE, insertcolor=TEXT,
                            padding=6, arrowcolor=MUTED)
            style.map(name, fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
                      bordercolor=[("focus", ACCENT)])
        style.configure("TSeparator", background=EDGE)
        style.configure("Vertical.TScrollbar", background=FIELD, troughcolor=PANEL,
                        bordercolor=EDGE, arrowcolor=MUTED, lightcolor=EDGE, darkcolor=EDGE)
        style.configure("Run.Horizontal.TProgressbar", background=ACCENT, troughcolor=FIELD,
                        bordercolor=FIELD, lightcolor=ACCENT, darkcolor=ACCENT, thickness=6)
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", TINT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", ACCENT)

    def layout(self):
        shell = ttk.Frame(self.root, padding=(20, 12), style="Shell.TFrame")
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell, style="Shell.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="EvoLab", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Treine agentes para resolver desafios", style="Subtitle.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        self.status = ttk.Label(header, text="●  Pronto para treinar", style="Status.TLabel")
        self.status.grid(row=0, column=1, rowspan=2, sticky="e", padx=(0, 9))
        self.open_button = ttk.Button(header, text="Abrir treino…", command=self.choose_run)
        self.open_button.grid(row=0, column=2, rowspan=2, sticky="e")

        body = ttk.Frame(shell, style="Shell.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=330)
        body.rowconfigure(0, weight=1)
        workspace = ttk.Frame(body, style="Shell.TFrame")
        workspace.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)
        self.layout_viewer(workspace)
        self.layout_chart(workspace)

        sidebar = ttk.Frame(body, padding=16, style="Card.TFrame")
        sidebar.grid(row=0, column=1, sticky="nsew")
        self.layout_settings(sidebar)
        self.error = ttk.Label(shell, text="", style="Error.TLabel", wraplength=820)
        self.error.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.error.grid_remove()
        shell.bind("<Configure>", lambda event: self.error.configure(wraplength=max(100, event.width - 62)))

    def layout_viewer(self, parent):
        viewer = ttk.Frame(parent, padding=11, style="Card.TFrame")
        viewer.grid(row=0, column=0, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)
        bar = ttk.Frame(viewer)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(bar, text="Acompanhe o desafio", style="Heading.TLabel").pack(side="left")
        self.mode = tk.StringVar(value=next(iter(MODES)))
        modes = ttk.Frame(bar)
        modes.pack(side="right")
        self.mode_buttons = []
        for label in MODES:
            button = ttk.Radiobutton(modes, text=label, value=label, variable=self.mode, command=self.change_view,
                                     style="Mode.TRadiobutton")
            button.pack(side="left", padx=(4, 0))
            self.mode_buttons.append(button)
        self.viewport = tk.Canvas(viewer, height=210, width=200, bg=STAGE, bd=0, highlightthickness=0)
        self.viewport.grid(row=1, column=0, sticky="nsew")
        self.viewport.bind("<Configure>", lambda _: self.draw_view())
        speed_bar = ttk.Frame(viewer)
        speed_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(speed_bar, text="Velocidade", style="Field.TLabel").pack(side="left", padx=(0, 8))
        self.speed = tk.StringVar(value="Normal")
        for label in SPEEDS:
            button = ttk.Radiobutton(speed_bar, text=label, value=label, variable=self.speed,
                                     command=self.change_speed, style="Mode.TRadiobutton")
            button.pack(side="left", padx=(0, 4))
            if label == "Headless":
                self.headless_button = button
        self.mode_help = ttk.Label(viewer, text=MODE_HELP["best"], style="Muted.TLabel")
        self.mode_help.grid(row=3, column=0, sticky="w", pady=(6, 0))
        viewer.bind("<Configure>", lambda event: self.mode_help.configure(wraplength=max(100, event.width - 28)))

    def layout_chart(self, parent):
        card = ttk.Frame(parent, padding=(11, 8), style="Card.TFrame")
        card.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        bar = ttk.Frame(card)
        bar.pack(fill="x", pady=(0, 2))
        ttk.Label(bar, text="Como o treino evolui", style="Heading.TLabel").pack(side="left")
        ttk.Label(bar, text="● Média", foreground=BLUE, font=("Segoe UI", 9)).pack(side="right")
        ttk.Label(bar, text="● Melhor", foreground=ACCENT, font=("Segoe UI", 9)).pack(side="right", padx=(0, 12))
        self.chart = tk.Canvas(card, height=104, width=200, bg=PANEL, bd=0, highlightthickness=0)
        self.chart.pack(fill="x")
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        self.chart.bind("<Motion>", self.chart_hover)
        self.chart.bind("<Leave>", lambda _: self.chart.delete("hover"))
        ttk.Separator(card).pack(fill="x", pady=(4, 6))
        scores = ttk.Frame(card)
        scores.pack(fill="x")
        self.metrics = []
        for column, (label, style) in enumerate((
            ("Melhor da rodada", "Best.Metric.TLabel"),
            ("Média dos agentes", "Mean.Metric.TLabel"),
            ("Recorde do treino", "Metric.TLabel"),
        )):
            scores.columnconfigure(column, weight=1, uniform="scores")
            cell = ttk.Frame(scores)
            cell.grid(row=0, column=column, sticky="w")
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")
            value = ttk.Label(cell, text="—", style=style)
            value.pack(anchor="w")
            self.metrics.append(value)

    def layout_settings(self, sidebar):
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)
        self.settings_title = ttk.Label(sidebar, text="Novo treino", style="Heading.TLabel")
        self.settings_title.grid(row=0, column=0, sticky="w", pady=(0, 14))

        pages = ttk.Frame(sidebar)
        pages.grid(row=1, column=0, sticky="nsew")
        pages.columnconfigure(0, weight=1)
        self.experiment = ttk.Frame(pages)
        self.experiment.grid(row=0, column=0, sticky="ew")
        self.advanced = ttk.Frame(pages)
        self.advanced.grid(row=0, column=0, sticky="ew")
        self.advanced.grid_remove()
        self.archive_page = ttk.Frame(pages)
        self.archive_page.grid(row=0, column=0, sticky="ew")
        self.archive_page.grid_remove()
        self.archive_title = ttk.Label(self.archive_page, style="Heading.TLabel", wraplength=275)
        self.archive_title.pack(anchor="w")
        self.archive_summary = ttk.Label(self.archive_page, style="Muted.TLabel", wraplength=275)
        self.archive_summary.pack(anchor="w", pady=(5, 12))
        self.archive_file = ttk.Label(self.archive_page, style="Muted.TLabel", wraplength=275)
        self.archive_file.pack(anchor="w")
        self.archive_actions = ttk.Frame(self.archive_page)
        self.archive_actions.pack(fill="x", pady=(18, 0))
        ttk.Button(self.archive_actions, text="Assistir à melhor rede", style="Start.TButton",
                   command=self.replay_run).pack(fill="x")
        ttk.Button(self.archive_actions, text="Continuar treino…",
                   command=self.continue_run).pack(fill="x", pady=(8, 0))
        ttk.Button(self.archive_actions, text="Novo treino", style="Secondary.TButton",
                   command=self.new_run).pack(fill="x", pady=(8, 0))

        parent = self.experiment
        self.path = tk.StringVar()
        self.arch_path = tk.StringVar()
        self.environment_choice = tk.StringVar(value=DEMO_ENV)
        self.labeled_choice(parent, "Desafio", self.environment_choice, (*CHALLENGES, FILE_ENV), self.change_environment)
        self.environment_details = ttk.Frame(parent)
        self.environment_details.pack(fill="x")
        self.demo_tip = ttk.Label(self.environment_details, style="Muted.TLabel", wraplength=275)
        self.environment_row = self.path_row(self.environment_details, self.path, self.pick_environment)

        self.template = tk.StringVar(value=next(iter(PRESETS)))
        self.labeled_choice(parent, "Rede dos agentes", self.template,
                            (*PRESETS, JSON_ARCH, PYTHON_ARCH), self.change_architecture, pady=(16, 0))
        self.arch_details = ttk.Frame(parent)
        self.arch_details.pack(fill="x")
        self.arch_summary = ttk.Label(self.arch_details, style="Muted.TLabel", wraplength=275)
        self.arch_row = self.path_row(self.arch_details, self.arch_path, self.pick_architecture)
        self.editor_box = ttk.Frame(self.arch_details)
        self.editor = tk.Text(self.editor_box, width=25, height=5, wrap="none", bg=FIELD, fg=TEXT,
                              insertbackground=TEXT, selectbackground=TINT, selectforeground=TEXT,
                              font=("Consolas", 10), relief="flat", padx=8, pady=3, highlightthickness=1,
                              highlightbackground=EDGE, highlightcolor=ACCENT, undo=True)
        scrollbar = ttk.Scrollbar(self.editor_box, orient="vertical", command=self.editor.yview)
        self.editor.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.editor.pack(side="left", fill="both", expand=True)
        self.change_architecture()

        ttk.Separator(parent).pack(fill="x", pady=(17, 10))
        self.plan = ttk.Label(parent, style="Muted.TLabel")
        self.plan.pack(anchor="w", pady=(0, 7))
        self.advanced_button = ttk.Button(parent, text="Ajustes avançados  ›", style="Secondary.TButton",
                                          command=lambda: self.show_settings(advanced=True, focus=True))
        self.advanced_button.pack(fill="x")

        defaults = Settings()
        self.entries = {}
        self.inputs = {}
        back = ttk.Button(self.advanced, text="‹  Voltar ao treino", style="Secondary.TButton",
                          command=lambda: self.show_settings(focus=True))
        back.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.back_button = back
        for row, (name, label, _) in enumerate(FIELDS, 1):
            ttk.Label(self.advanced, text=label, style="Field.TLabel").grid(
                row=row * 2 - 1, column=0, sticky="w", pady=(3, 0)
            )
            self.number_input(self.advanced, name, defaults).grid(
                row=row * 2 - 1, column=1, sticky="e", padx=(7, 0), pady=(3, 0)
            )
            if name in FIELD_HELP:
                ttk.Label(self.advanced, text=FIELD_HELP[name], style="Muted.TLabel", wraplength=275).grid(
                    row=row * 2, column=0, columnspan=2, sticky="w"
                )
        self.advanced.columnconfigure(0, weight=1)
        for name in ("population", "generations"):
            self.entries[name].trace_add("write", self.update_plan)
        self.update_plan()

        footer = ttk.Frame(sidebar)
        footer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.progress_group = ttk.Frame(footer)
        self.progress_text = ttk.Label(self.progress_group, style="Muted.TLabel")
        self.progress_text.pack(anchor="w")
        self.progress = ttk.Progressbar(self.progress_group, style="Run.Horizontal.TProgressbar", length=1, maximum=1)
        self.progress.pack(fill="x", pady=(5, 7))
        self.eval_time = ttk.Label(self.progress_group, style="Muted.TLabel")
        self.eval_time.pack(anchor="w", pady=(0, 9))
        self.buttons = ttk.Frame(footer)
        self.buttons.pack(fill="x")
        self.start_button = ttk.Button(self.buttons, text="Começar treino", style="Start.TButton", command=self.start)
        self.start_button.pack(fill="x")
        self.stop_button = ttk.Button(self.buttons, text="Parar treino", command=self.stop, padding=(12, 10))

    def show_settings(self, advanced=False, focus=False):
        visible = self.advanced if advanced else self.experiment
        hidden = self.experiment if advanced else self.advanced
        self.archive_page.grid_remove()
        hidden.grid_remove()
        visible.grid()
        self.settings_title.configure(text="Ajustes avançados" if advanced else "Novo treino")
        if focus:
            (self.back_button if advanced else self.advanced_button).focus_set()

    def update_plan(self, *_):
        try:
            agents = int(self.entries["population"].get())
            rounds = int(self.entries["generations"].get())
        except ValueError:
            agents = rounds = 0
        self.plan.configure(text=f"{agents} agentes · {rounds} rodadas" if agents > 0 and rounds > 0
                            else "Revise o número de agentes e de rodadas.")

    def number_input(self, parent, name, defaults):
        value = tk.StringVar(value=str(getattr(defaults, name)))
        widget = ttk.Entry(parent, width=9, textvariable=value)
        self.entries[name] = value
        self.inputs[name] = widget
        return widget

    @staticmethod
    def labeled_choice(parent, label, value, options, handler, pady=(0, 0)):
        group = ttk.Frame(parent)
        group.pack(fill="x", pady=pady)
        ttk.Label(group, text=label, style="Field.TLabel").pack(anchor="w", pady=(0, 3))
        widget = ttk.Combobox(group, textvariable=value, values=options, state="readonly", width=18)
        widget.pack(fill="x")
        widget.bind("<<ComboboxSelected>>", handler)

    @staticmethod
    def path_row(parent, value, pick):
        row = ttk.Frame(parent)
        ttk.Entry(row, textvariable=value, width=15).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Selecionar…", command=pick).pack(side="left", padx=(5, 0))
        return row

    def change_environment(self, _=None):
        self.demo_tip.pack_forget()
        self.environment_row.pack_forget()
        challenge = self.environment_choice.get()
        if challenge in CHALLENGES:
            self.demo_tip.configure(text=CHALLENGES[challenge][1])
            self.demo_tip.pack(anchor="w", pady=(4, 0))
        else:
            self.environment_row.pack(fill="x", pady=(4, 0))
        if self.session is None:
            self.refresh_preview()

    def show_error(self, message=""):
        self.error.configure(text=message)
        if message:
            self.error.grid()
        else:
            self.error.grid_remove()

    def set_status(self, message, style="Status.TLabel"):
        self.status.configure(text="●  " + message, style=style)

    def pick_environment(self):
        path = filedialog.askopenfilename(parent=self.root, title="Selecionar ambiente externo",
                                           filetypes=[("Ambientes Python ou executáveis", "*.py *.exe"), ("Todos os arquivos", "*")])
        if path:
            self.path.set(path)

    def pick_architecture(self):
        path = filedialog.askopenfilename(parent=self.root, title="Selecionar arquitetura Python",
                                           filetypes=[("Arquivos Python", "*.py")])
        if path:
            self.arch_path.set(path)

    def change_architecture(self, _=None):
        selected = self.template.get()
        previous = self.previous_architecture
        if previous == JSON_ARCH:
            self.custom_json = self.editor.get("1.0", "end-1c")
        self.previous_architecture = selected
        self.arch_row.pack_forget()
        self.arch_summary.pack_forget()
        self.editor_box.pack_forget()
        if selected == PYTHON_ARCH:
            self.arch_row.pack(fill="x", pady=(4, 0))
        elif selected == JSON_ARCH:
            if self.custom_json is None:
                template, _ = PRESETS.get(previous, next(iter(PRESETS.values())))
                config = TEMPLATES[template]
                lines = [f'  "{key}": {json.dumps(value)}' for key, value in config.items()]
                self.custom_json = "{\n" + ",\n".join(lines) + "\n}"
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", self.custom_json)
            self.editor.edit_reset()
            self.editor_box.pack(fill="x", pady=(4, 0))
        else:
            _, description = PRESETS[selected]
            self.arch_summary.configure(text=description)
            self.arch_summary.pack(anchor="w", pady=(4, 0))

    def start(self):
        if self.session is not None:
            return
        self.show_error()
        values = {}
        for name, label, cast in FIELDS:
            raw = self.entries[name].get().strip()
            try:
                values[name] = cast(raw.replace(",", ".") if cast is float else raw)
            except ValueError:
                self.show_settings(advanced=True)
                self.inputs[name].focus_set()
                self.inputs[name].selection_range(0, "end")
                self.show_error(f"Revise o valor de {label.lower()}.")
                return
        try:
            settings = Settings(**values)
            settings.validate()
        except ValueError as error:
            self.show_settings(advanced=True)
            self.show_error(str(error))
            return
        choice = self.environment_choice.get()
        path = self.path.get().strip() if choice == FILE_ENV else CHALLENGES[choice][0]
        if choice == FILE_ENV and not self.check_path(path, "do ambiente"):
            return
        selected = self.template.get()
        python = selected == PYTHON_ARCH
        if python:
            source = self.arch_path.get().strip()
            if not self.check_path(source, "da arquitetura"):
                return
        elif selected == JSON_ARCH:
            try:
                source = json.loads(self.editor.get("1.0", "end"))
            except json.JSONDecodeError as error:
                self.show_settings()
                self.editor.focus_set()
                self.editor.mark_set("insert", f"{error.lineno}.{error.colno - 1}")
                self.editor.see("insert")
                self.show_error(f"Revise o JSON na linha {error.lineno}, coluna {error.colno}.")
                return
        else:
            source = TEMPLATES[PRESETS[selected][0]]
        self.show_settings()
        self.clear_frames()
        self.latest = None
        self.chart_data = []
        self.draw_chart()
        for metric in self.metrics:
            metric.configure(text="—")
        self.total = settings.generations
        self.progress.configure(maximum=self.total, value=0)
        self.progress_text.configure(text=f"0 de {self.total} rodadas")
        self.eval_time.configure(text="")
        self.progress_group.pack(fill="x", before=self.buttons)
        experiment = Experiment.create(path, source, python=python)
        self.begin_session(Session(experiment, settings, self.requested_view()))
        self.change_view()

    def begin_session(self, session):
        self.session = session
        self.open_button.configure(state="disabled")
        self.archive_actions.pack_forget()
        had_focus = self.start_button == self.root.focus_get()
        self.start_button.pack_forget()
        self.stop_button.configure(state="normal")
        self.stop_button.pack(fill="x")
        if had_focus:
            self.stop_button.focus_set()

    def choose_run(self):
        if self.session:
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Abrir treino anterior",
            initialdir=DEFAULT_RUNS if DEFAULT_RUNS.is_dir() else DEFAULT_RUNS.parent,
            filetypes=[("Treinos EvoLab", "*.evolab")],
        )
        if not path:
            return
        try:
            saved = load_run(path)
        except Exception as error:
            self.show_error(f"Não foi possível abrir o treino. {error}")
            return
        self.show_error()
        self.clear_frames()
        self.show_saved(saved)
        self.set_status("Treino carregado", "Active.Status.TLabel")
        self.view_message = ("Treino anterior", "Escolha assistir à melhor rede ou continuar de onde parou.")
        self.mode_help.configure(text="Os resultados abaixo são do treino salvo.")
        self.draw_view()

    def show_saved(self, saved):
        self.saved_run = saved
        self.experiment.grid_remove()
        self.advanced.grid_remove()
        self.archive_page.grid()
        self.settings_title.configure(text="Treino salvo")
        challenge = next((label for label, (key, _) in CHALLENGES.items()
                          if key == saved.experiment.environment), Path(saved.experiment.environment).name)
        self.archive_title.configure(text=challenge)
        count = saved.state.completed
        rounds = "1 rodada" if count == 1 else f"{count} rodadas"
        self.archive_summary.configure(text=f"{saved.settings.population} agentes · {rounds}\n"
                                           f"Melhor pontuação na última rodada  {saved.state.rows[-1][1]:.2f}")
        self.archive_file.configure(text=saved.path.name)
        self.archive_actions.pack(fill="x", pady=(18, 0))
        self.start_button.pack_forget()
        self.progress_group.pack_forget()
        self.total = count
        self.chart_data = [row[:3] for row in saved.state.rows[-180:]]
        self.update_metrics(saved.state.rows[-1][:5])
        self.draw_chart()

    def new_run(self):
        self.saved_run = None
        self.latest = None
        self.clear_frames()
        self.chart_data = []
        for metric in self.metrics:
            metric.configure(text="—")
        self.draw_chart()
        self.show_settings()
        self.show_error()
        self.set_status("Pronto para treinar")
        self.start_button.pack(fill="x")
        self.refresh_preview()

    def replay_run(self):
        if self.session or self.saved_run is None:
            return
        self.show_error()
        self.clear_frames()
        self.mode.set(next(iter(MODES)))
        if self.speed.get() == "Headless":
            self.speed.set("Normal")
        self.headless_button.configure(state="disabled")
        for button in self.mode_buttons:
            button.configure(state="disabled")
        saved = self.saved_run
        self.begin_session(Session(saved.experiment, saved.settings, "best", saved=saved, replay=True))
        self.set_status("Reproduzindo rede salva", "Active.Status.TLabel")
        self.view_message = ("Preparando a reprodução", "A melhor rede da última rodada vai jogar novamente.")
        self.mode_help.configure(text="Reprodução da melhor rede do treino salvo.")
        self.draw_view()

    def continue_run(self):
        if self.session or self.saved_run is None:
            return
        rounds = simpledialog.askinteger("Continuar treino", "Quantas rodadas você quer acrescentar?",
                                         parent=self.root, minvalue=1, initialvalue=25)
        if rounds is None:
            return
        self.show_error()
        self.clear_frames()
        saved = self.saved_run
        self.latest = saved.state.rows[-1][:5]
        self.total = saved.state.completed + rounds
        self.progress.configure(maximum=self.total, value=saved.state.completed)
        self.progress_text.configure(text=f"{saved.state.completed} de {self.total} rodadas")
        self.progress_group.pack(fill="x", before=self.buttons)
        self.begin_session(Session(saved.experiment, rounds, self.requested_view(), saved=saved))
        self.change_view()

    def update_metrics(self, values):
        number, best, mean, record, seconds = values
        for metric, value in zip(self.metrics, (best, mean, record)):
            metric.configure(text=f"{value:.2f}")
        self.progress.configure(value=number)
        self.progress_text.configure(text=f"{number} de {self.total} rodadas")
        self.eval_time.configure(text=f"Última avaliação · {seconds * 1000:.1f} ms")

    def check_path(self, path, role):
        if path and Path(path).is_file():
            return True
        self.show_settings()
        self.show_error(f"Selecione um arquivo válido {role}.")
        return False

    def stop(self):
        if self.session:
            self.session.cancel()
            self.clear_frames()
            self.stop_button.configure(state="disabled")
            title = "Encerrando reprodução" if self.session.replay else "Encerrando treino"
            self.set_status(title)
            self.view_message = (title, "Aguardando o desafio terminar a rodada atual.")
            self.draw_view()

    def close(self):
        if self.session:
            self.closing = True
            self.stop()
        else:
            self.root.destroy()

    def requested_view(self):
        return "none" if self.speed.get() == "Headless" else MODES[self.mode.get()]

    def change_speed(self):
        session = self.session
        if session and (session.replay or session.get_view() == self.requested_view()):
            if self.frame_job is not None:
                self.schedule_frame()
        else:
            self.change_view()

    def change_view(self):
        view = self.requested_view()
        self.mode_help.configure(text=MODE_HELP[view])
        if self.session is None:
            if self.saved_run:
                return
            self.refresh_preview()
            return
        if self.session.stop.is_set() or self.session.replay:
            return
        if self.session.get_view() == view and self.current_frame is not None:
            return
        self.clear_frames()
        if self.session.get_view() != view:
            self.session.set_view(view)
        if view == "none":
            self.set_status("Modo headless ativo", "Active.Status.TLabel")
            self.view_message = ("Modo headless ativo", "As pontuações continuam aparecendo abaixo.")
        else:
            self.set_status("Treinando", "Active.Status.TLabel")
            self.view_message = ("Treino em andamento", "Os agentes aparecem assim que a rodada terminar.")
        self.draw_view()

    def refresh_preview(self):
        self.current_frame = None
        view = self.requested_view()
        self.mode_help.configure(text=MODE_HELP[view])
        if view == "none":
            self.view_message = ("Treino sem animação", "Comece um treino para acompanhar as pontuações ao vivo.")
        elif self.environment_choice.get() in CHALLENGES:
            key, description = CHALLENGES[self.environment_choice.get()]
            self.current_frame = BUILTINS[key].preview()
            self.mode_help.configure(text=f"Prévia do desafio · {description}")
        else:
            self.view_message = ("Importe um desafio", "Selecione um arquivo e comece o treino para ver os agentes.")
        self.draw_view()

    def poll(self):
        session = self.session
        if session:
            latest, frames, done, error = session.snapshot()
            if latest is not None and latest != self.latest:
                self.latest = latest
                self.update_metrics(latest)
                with session.lock:
                    self.chart_data = list(session.history)
                self.draw_chart()
            if frames:
                self.frames = frames
                self.frame_index = 0
                self.next_frame()
            if done:
                if self.closing and not error:
                    self.root.destroy()
                    return
                self.closing = False
                self.session = None
                self.stop_button.pack_forget()
                self.open_button.configure(state="normal")
                if session.saved:
                    self.show_saved(session.saved)
                elif self.saved_run:
                    self.show_saved(self.saved_run)
                else:
                    self.start_button.pack(fill="x")
                for button in self.mode_buttons:
                    button.configure(state="normal")
                self.headless_button.configure(state="normal")
                if error:
                    self.set_status("Falha na reprodução" if session.replay else "Falha no treino",
                                    "Failed.Status.TLabel")
                    self.show_error(error)
                elif session.replay:
                    self.set_status("Reprodução interrompida" if session.stop.is_set() else "Reprodução concluída")
                elif session.stop.is_set():
                    self.set_status("Treino interrompido e salvo" if session.saved else "Treino interrompido")
                else:
                    self.set_status("Treino concluído e salvo", "Active.Status.TLabel")
                if self.current_frame is None:
                    if self.saved_run:
                        self.view_message = ("Treino anterior", "Assista à melhor rede ou continue o treino.")
                    else:
                        self.view_message = ("Treino encerrado", "Confira as pontuações abaixo.")
                    self.draw_view()
        self.root.after(35, self.poll)

    def clear_frames(self):
        if self.frame_job is not None:
            self.root.after_cancel(self.frame_job)
            self.frame_job = None
        self.current_frame = None
        self.frames = []
        self.frame_motion = {}
        self.motion_items = []

    def next_frame(self):
        self.frame_job = None
        if self.frame_index == len(self.frames):
            self.frames = []
            self.session.played.set()
            return
        self.current_frame = self.frames[self.frame_index]
        self.frame_index += 1
        try:
            self.frame_duration_ms = max(1, int(self.current_frame.get("duration_ms", 33)))
            following = self.frames[self.frame_index] if self.frame_index < len(self.frames) else None
            self.interpolated_clip, self.frame_motion = motion_between(self.current_frame, following)
            self.render(self.current_frame)
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError, tk.TclError) as error:
            self.show_error(f"Quadro inválido. {error}")
            self.view_message = ("Quadro inválido", "Confira o formato de visualização do ambiente.")
            self.clear_frames()
            self.draw_view()
            self.session.played.set()
            return
        self.frame_started = monotonic()
        self.schedule_frame()

    def schedule_frame(self):
        if self.frame_job is not None:
            self.root.after_cancel(self.frame_job)
        self.frame_job = None
        self.animate_frame()

    def animate_frame(self):
        self.frame_job = None
        speed = self.speed.get()
        multiplier = SPEEDS[speed]
        if speed == "Normal" and not self.interpolated_clip:
            multiplier = 1.5
        duration = self.frame_duration_ms * multiplier
        elapsed = (monotonic() - self.frame_started) * 1000
        remaining = max(1, round(duration - elapsed))
        if speed == "Normal" and self.motion_items and remaining > 1:
            self.move_shapes(elapsed / duration)
            if remaining > ANIMATION_TICK_MS:
                self.frame_job = self.root.after(ANIMATION_TICK_MS, self.animate_frame)
                return
        self.frame_job = self.root.after(remaining, self.next_frame)

    def move_shapes(self, fraction):
        delta = fraction - self.motion_fraction
        for item, dx, dy in self.motion_items:
            self.viewport.move(item, dx * delta, dy * delta)
        self.motion_fraction = fraction

    def draw_view(self):
        if self.current_frame is not None:
            fraction = self.motion_fraction
            self.render(self.current_frame)
            if fraction:
                self.move_shapes(fraction)
            return
        canvas = self.viewport
        canvas.delete("all")
        canvas.configure(bg=STAGE)
        x, y = canvas.winfo_width() / 2, canvas.winfo_height() / 2
        title, subtitle = self.view_message
        width = max(100, canvas.winfo_width() - 36)
        canvas.create_text(x, y - 13, text=title, fill=STAGE_TEXT, font=("Segoe UI", 17, "bold"),
                           width=width, justify="center")
        canvas.create_text(x, y + 20, text=subtitle, fill=STAGE_MUTED, font=("Segoe UI", 10),
                           width=width, justify="center")

    def render(self, frame):
        canvas = self.viewport
        canvas.delete("all")
        self.motion_items = []
        self.motion_fraction = 0.0
        canvas.configure(bg=frame.get("background", STAGE))
        width, height = frame["width"], frame["height"]
        if width <= 0 or height <= 0:
            raise ValueError("width e height devem ser positivos.")
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        scale = min(cw / width, ch / height)
        left, top = (cw - width * scale) / 2, (ch - height * scale) / 2
        for shape in frame["shapes"]:
            kind = shape["kind"]
            fill, outline = shape.get("fill", ""), shape.get("outline", "")
            if kind == "line":
                points = [point * scale + (left if i % 2 == 0 else top)
                          for i, point in enumerate(shape["points"])]
                if len(points) < 4 or len(points) % 2:
                    raise ValueError("Uma linha precisa de pelo menos dois pares de coordenadas.")
                canvas.create_line(*points, fill=fill or STAGE_TEXT, width=max(1, shape.get("width", 1) * scale))
            else:
                x, y = left + shape["x"] * scale, top + shape["y"] * scale
                if kind == "circle":
                    r = shape["r"] * scale
                    item = canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline)
                elif kind == "rect":
                    item = canvas.create_rectangle(x, y, x + shape["w"] * scale, y + shape["h"] * scale,
                                                   fill=fill, outline=outline)
                else:
                    raise ValueError(f"Forma desconhecida: {kind}")
                key = shape.get("id")
                if isinstance(key, (str, int)) and key in self.frame_motion:
                    dx, dy = self.frame_motion[key]
                    self.motion_items.append((item, dx * scale, dy * scale))

    def draw_chart(self):
        canvas = self.chart
        width, height = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("all")
        self.chart_geometry = None
        if not self.chart_data:
            for percent in (0.2, 0.5, 0.8):
                canvas.create_line(0, height * percent, width, height * percent, fill=FIELD)
            canvas.create_text(width / 2, height / 2, text="As pontuações aparecem aqui depois da primeira rodada.",
                               fill=MUTED, font=("Segoe UI", 9))
            return
        generations, best, means = zip(*self.chart_data)
        low, high = min(min(best), min(means)), max(max(best), max(means))
        if high == low:
            delta = max(1, abs(low) * 0.02)
            low -= delta
            high += delta
        left, right, top, bottom = 53, max(54, width - 12), 13, height - 19
        for fraction in (0, 0.5, 1):
            y = bottom - fraction * (bottom - top)
            canvas.create_line(left, y, right, y, fill=EDGE)
            canvas.create_text(left - 6, y, text=f"{low + fraction * (high - low):.3g}", fill=MUTED,
                               anchor="e", font=("Segoe UI", 8))
        positions = [(left + right) / 2] if len(generations) == 1 else [
            left + i * (right - left) / (len(generations) - 1) for i in range(len(generations))
        ]
        tick_indices = {0, (len(generations) - 1) // 2, len(generations) - 1}
        for i in sorted(tick_indices):
            canvas.create_text(positions[i], height - 7, text=str(generations[i]), fill=MUTED,
                               font=("Segoe UI", 8))
        for color, series in ((BLUE, means), (ACCENT, best)):
            points = []
            for x, value in zip(positions, series):
                y = bottom - (value - low) / (high - low) * (bottom - top)
                points.extend((x, y))
            if len(points) > 2:
                canvas.create_line(*points, width=2, fill=color)
            x, y = points[-2:]
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline=PANEL)
        self.chart_geometry = positions, top, bottom

    def chart_hover(self, event):
        canvas = self.chart
        canvas.delete("hover")
        if self.chart_geometry is None:
            return
        positions, top, bottom = self.chart_geometry
        if not top <= event.y <= bottom:
            return
        index = min(range(len(positions)), key=lambda i: abs(positions[i] - event.x))
        x = positions[index]
        generation, best, mean = self.chart_data[index]
        canvas.create_line(x, top, x, bottom, fill=MUTED, dash=(2, 3), tags="hover")
        anchor = "ne" if x > canvas.winfo_width() / 2 else "nw"
        distance = -8 if anchor == "ne" else 8
        label = canvas.create_text(x + distance, top + 4, anchor=anchor, fill=TEXT, tags="hover",
                                   text=f"Rodada {generation}\nMelhor {best:.2f}   ·   Média {mean:.2f}",
                                   font=("Segoe UI", 9))
        left, upper, right, lower = canvas.bbox(label)
        box = canvas.create_rectangle(left - 5, upper - 4, right + 5, lower + 4,
                                      fill=PANEL, outline=EDGE, tags="hover")
        canvas.tag_raise(label, box)


def launch():
    root = tk.Tk()
    App(root)
    root.mainloop()
