"""Desktop launcher and headless entry point."""

import argparse
import signal
import sys
from pathlib import Path
from threading import Event
from time import monotonic

from environments import BUILTINS
from evolution import Settings, TrainingState, train
from network import TEMPLATES
from runs import DEFAULT_RUNS, Experiment, load_run, save_run


def main():
    parser = argparse.ArgumentParser(description="EvoLab — treino evolutivo de redes neurais")
    parser.add_argument("--headless", action="store_true", help="treina sem abrir uma janela")
    challenge = parser.add_mutually_exclusive_group()
    challenge.add_argument("--environment", help="arquivo .py ou executável do ambiente")
    challenge.add_argument("--challenge", choices=BUILTINS, help="desafio integrado")
    parser.add_argument("--external", action="store_true", help="executa um ambiente .py pelo protocolo JSON")
    parser.add_argument("--resume", type=Path, help="retoma um treino salvo; --generations indica rodadas adicionais")
    parser.add_argument("--output-dir", type=Path, help="diretório dos arquivos de treino")
    architecture = parser.add_mutually_exclusive_group()
    architecture.add_argument("--template", choices=("compacta", "multicamadas", "linear"))
    architecture.add_argument("--architecture", type=Path, help="arquivo .json ou .py da rede")
    defaults = Settings()
    for name, default in vars(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=None)
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in vars(defaults) if getattr(args, name) is not None}
    if not args.headless:
        if any((args.environment, args.challenge, args.external, args.resume, args.output_dir,
                args.template, args.architecture, overrides)):
            parser.error("As opções de treino no terminal exigem --headless.")
        from app import launch
        launch()
        return 0

    if args.resume and (any((args.environment, args.challenge, args.external, args.template, args.architecture))
                        or overrides.keys() - {"generations"}):
        parser.error("--resume usa o desafio, a rede e os ajustes salvos. Use --generations para acrescentar rodadas.")
    templates = dict(zip(("compacta", "multicamadas", "linear"), TEMPLATES.values()))
    try:
        state = TrainingState()
        if args.resume:
            saved = load_run(args.resume)
            rounds = defaults.generations if args.generations is None else args.generations
            if rounds < 1:
                raise ValueError("Acrescente pelo menos uma rodada ao treino.")
            settings, state = saved.continuation(rounds)
            experiment = saved.experiment
        else:
            settings = Settings(**overrides)
            python = bool(args.architecture and args.architecture.suffix.lower() == ".py")
            source = templates[args.template or "compacta"]
            if args.architecture:
                source = args.architecture if python else args.architecture.read_text(encoding="utf-8")
            experiment = Experiment.create(args.environment or args.challenge, source,
                                           python=python, external=args.external)
        settings.validate()
        first = state.completed + 1
        stop = Event()
        outcome = "completed"
        previous = signal.signal(signal.SIGINT, lambda *_: stop.set())
        try:
            with experiment.open() as (environment, model):
                print(f"Modo acelerado ativo | {settings.population} indivíduos | "
                      f"rodadas {first} a {settings.generations}", flush=True)
                last_report = monotonic()

                def report(generation):
                    nonlocal last_report
                    now = monotonic()
                    if generation.number in (first, settings.generations) or now - last_report >= 1:
                        print(f"Geração {generation.number}/{settings.generations} | melhor {generation.best:.3f} | "
                              f"média {generation.mean:.3f} | recorde {generation.record:.3f}", flush=True)
                        last_report = now

                train(environment, model, settings, lambda: "none", stop, report, state=state)
            if stop.is_set():
                outcome = "interrupted"
        except Exception:
            outcome = "failed"
            raise
        finally:
            try:
                if state.completed >= first:
                    saved = save_run(experiment, settings, state, outcome, args.output_dir or DEFAULT_RUNS)
                    print(f"Treino salvo em {saved.path}", flush=True)
            finally:
                signal.signal(signal.SIGINT, previous)
        if outcome == "interrupted":
            print("Treino interrompido.", file=sys.stderr)
            return 130
    except KeyboardInterrupt:
        print("Treino interrompido.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
