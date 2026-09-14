import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def ask(text, default=""):
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip().strip('"')
    return value or default


def run(script, *arguments):
    command = [PYTHON, str(ROOT / script), *map(str, arguments)]
    print("\nЗапуск:", " ".join(f'"{part}"' if " " in part else part for part in command), "\n")
    result = subprocess.run(command, cwd=ROOT)
    print(f"\nЗавершено с кодом {result.returncode}. Нажмите Enter, чтобы вернуться в меню.")
    input()


def prepare():
    archive = ask("Путь к data.zip")
    if not archive:
        print("Путь не задан.")
        return
    trip = ask("Поездка", "trip01")
    limit = ask("Количество кадров, 0 = все", "0")
    output = ask("Каталог датасета", "dataset")
    run("prepare_data.py", archive, "--trip", trip, "--limit", limit, "--out", output)


def label():
    data = Path(ask("Каталог поездки", "dataset/trip01"))
    run("label.py", data / "images", data / "masks")


def train():
    data = ask("Каталог поездки", "dataset/trip01")
    epochs = ask("Количество эпох", "30")
    batch = ask("Размер батча", "4")
    output = ask("Куда сохранить лучшую модель", "runs/roadnet.pt")
    pretrained = ask("Использовать предобученный ResNet-50? y/n", "y").lower()
    arguments = [data, "--epochs", epochs, "--batch", batch, "--out", output]
    if pretrained in {"n", "no", "нет"}:
        arguments.append("--no-pretrained")
    run("train.py", *arguments)


def infer():
    source = ask("Путь к видео или каталогу изображений")
    if not source:
        print("Источник не задан.")
        return
    checkpoint = ask("Путь к модели", "runs/roadnet.pt")
    output = ask("Каталог результата", "runs/predictions")
    threshold = ask("Порог дороги", "0.5")
    show = ask("Показывать кадры в реальном времени? y/n", "y").lower()
    arguments = [source, checkpoint, "--out", output, "--threshold", threshold]
    if show in {"n", "no", "нет"}:
        arguments.append("--no-show")
    run("infer.py", *arguments)


def menu():
    actions = {"1": prepare, "2": label, "3": train, "4": infer, "5": lambda: run("test_corridor.py")}
    while True:
        print(
            "\n=== AI ROAD — ДОРОГА В КАРЬЕРЕ ===\n"
            "1. Подготовить датасет из ZIP\n"
            "2. Открыть инструмент разметки\n"
            "3. Обучить нейросеть\n"
            "4. Обработать видео/кадры\n"
            "5. Проверить геометрию зон\n"
            "0. Выход\n"
        )
        choice = input("Выберите пункт: ").strip()
        if choice == "0":
            break
        action = actions.get(choice)
        if action:
            action()
        else:
            print("Нет такого пункта.")


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
