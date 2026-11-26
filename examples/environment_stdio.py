"""Small protocol reference. Agents try to predict an input with one action."""

import json
import random
import sys


def send(message):
    print(json.dumps(message, allow_nan=False), flush=True)


def evaluate(request):
    rng = random.Random(request["seed"])
    weights = request["weights"]
    target = rng.uniform(-1, 1)
    send({"type": "predict", "observations": [[target] for _ in weights]})
    actions = json.loads(sys.stdin.readline())["actions"]
    scores = [-abs(target - action[0]) for action in actions]
    response = {"type": "result", "scores": scores}
    if request["view"] != "none":
        best = max(range(len(scores)), key=scores.__getitem__)
        indices = [best] if request["view"] == "best" else range(len(scores))
        shapes = [{"kind": "line", "points": [30, 79, 350, 79], "fill": "#7688a8"},
                  {"kind": "circle", "x": 190 + target * 160, "y": 79, "r": 9, "fill": "#78edb7"}]
        for i in indices:
            action = max(-1, min(1, actions[i][0]))
            shapes.append({"kind": "circle", "x": 190 + action * 160, "y": 96 + i % 4 * 5,
                           "r": 4, "fill": "#89a7da"})
        response["frames"] = [{"width": 380, "height": 160, "duration_ms": 120, "shapes": shapes}]
    send(response)


if __name__ == "__main__":
    for line in sys.stdin:
        request = json.loads(line)
        if request["type"] == "describe" and request["version"] == 1:
            send({"type": "description", "inputs": 1, "outputs": 1})
        elif request["type"] == "evaluate":
            evaluate(request)
        else:
            send({"type": "error", "message": "Mensagem de protocolo incompatível."})
