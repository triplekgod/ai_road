"""Interactive console menu; all actions use the same public CLI."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def ask(label, default=None, required=True):
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip().strip('"') or default
    if required and not value:
        print("Поле обязательно. Операция отменена.")
    return value


def run(script, arguments):
    command = [sys.executable, str(ROOT / script), *map(str, arguments)]
    print("\nЗапуск:", subprocess.list2cmdline(command), "\n")
    return subprocess.run(command, check=False).returncode


def prepare():
    video = ask("Путь к видео")
    output = ask("Папка кадров одного рейса", "data/trip01/images")
    group = ask("Уникальный идентификатор рейса", "trip01")
    every = ask("Сохранять каждый N-й кадр", "10")
    if all((video, output, group, every)):
        run("prepare.py", [video, output, "--group", group, "--every", every])


def annotate():
    images = ask("Папка кадров", "data/trip01/images")
    masks = ask("Папка масок", "data/trip01/masks")
    if images and masks:
        run("annotate.py", [images, masks])


def split_data():
    groups = ask("JSON со списком рейсов", "groups.json")
    output = ask("Выходной manifest", "splits.json")
    if groups and output:
        run("data_tools.py", ["split", "--groups", groups, "--output", output])


def train(mode="new"):
    manifest = ask("Manifest с train/validation/test", "splits.json")
    epochs = ask("Итоговое число эпох" if mode == "resume" else "Количество эпох", "50")
    batch = ask("Размер batch", "8")
    output = ask("Путь к лучшей модели", "models/best.pth")
    if not all((manifest, epochs, batch, output)):
        return
    args = ["--manifest", manifest, "--epochs", epochs, "--batch-size", batch, "--output", output]
    if mode == "new":
        architecture = ask("Архитектура: lite_road_net / fast_scnn_lite", "lite_road_net")
        width = ask("Ширина входа (кратно 16)", "256")
        height = ask("Высота входа (кратно 16)", "144")
        roi = ask("Исключить верхнюю долю кадра", "0.20")
        if not all((architecture, width, height, roi)):
            return
        args += ["--architecture", architecture, "--input-width", width, "--input-height", height, "--roi-top", roi]
    else:
        checkpoint = ask("Checkpoint (.last.pth для resume)")
        if not checkpoint:
            return
        args += ["--" + mode, checkpoint]
    run("train.py", args)


def infer(benchmark=False):
    video = ask("Путь к видео")
    model = ask("Путь к модели .pth / .onnx")
    backend = ask("Backend: pytorch / onnxruntime / openvino", "pytorch")
    output = ask("Результат MP4", "results/road_zones.mp4")
    report = ask("Отчет JSON", "results/benchmark.json")
    geometry = ask("Геометрия: centerline / legacy", "centerline")
    if all((video, model, backend, output, report, geometry)):
        args = [video, model, "--backend", backend, "--output", output, "--report", report, "--geometry", geometry]
        if benchmark:
            args.append("--no-display")
        run("infer.py", args)


def evaluate():
    model = ask("Путь к модели")
    backend = ask("Backend", "pytorch")
    manifest = ask("Manifest", "splits.json")
    output = ask("Отчет", "results/evaluation.json")
    if all((model, backend, manifest, output)):
        run("evaluate.py", [model, "--manifest", manifest, "--split", "test", "--backend", backend, "--output", output])


def export():
    model = ask("Checkpoint .pth")
    output = ask("Выходной ONNX", "models/road.onnx")
    if model and output:
        run("export.py", ["--checkpoint", model, "--output", output, "--check"])


def quantize():
    model = ask("Модель FP32 ONNX")
    manifest = ask("Manifest (200–500 кадров калибровки из train)", "splits.json")
    output = ask("Модель INT8", "models/road.int8.onnx")
    if model and manifest and output:
        run("quantize.py", ["--model", model, "--manifest", manifest, "--output", output, "--samples", "200"])


def menu():
    actions = {
        "1": ("Извлечь BMP-кадры", prepare),
        "2": ("Разметить кадры", annotate),
        "3": ("Разделить данные по рейсам", split_data),
        "4": ("Обучить новую модель", train),
        "5": ("Возобновить обучение с оптимизатором (resume)", lambda: train("resume")),
        "6": ("Дообучить только веса (finetune)", lambda: train("finetune")),
        "7": ("Обработать видео", infer),
        "8": ("Benchmark без окна", lambda: infer(True)),
        "9": ("Оценить на независимом test", evaluate),
        "10": ("Экспортировать ONNX и проверить соответствие", export),
        "11": ("Статическое INT8-квантование", quantize),
    }
    while True:
        print("\nАНАЛИЗ КАРЬЕРНОЙ ДОРОГИ")
        for key, (title, _) in actions.items():
            print(f"  {key}. {title}")
        print("  0. Выход")
        try:
            choice = input("\nВыберите действие: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if choice == "0":
            return
        if choice not in actions:
            print("Неизвестный пункт меню.")
            continue
        try:
            actions[choice][1]()
        except (ValueError, OSError) as error:
            print(f"Ошибка: {error}")


if __name__ == "__main__":
    menu()
