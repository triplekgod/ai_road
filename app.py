"""Interactive console menu for the quarry-road project."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def ask(label, default=None, required=True):
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip().strip('"')
    value = value or default
    if required and not value:
        print("Поле обязательно. Операция отменена.")
        return None
    return value


def run(script, arguments):
    command = [sys.executable, "-m", *map(str, arguments)] if script == "-m" else [sys.executable, str(ROOT / script), *map(str, arguments)]
    print("\nЗапуск:", " ".join(f'"{item}"' if " " in item else item for item in command), "\n")
    subprocess.run(command, check=False)


def prepare():
    video = ask("Путь к исходному видео")
    output = ask("Папка для BMP-кадров", "data\\road\\images")
    every = ask("Сохранять каждый N-й кадр", "10")
    if video and output and every: run("prepare.py", [video, output, "--every", every])


def annotate():
    images = ask("Папка с BMP-кадрами", "data\\road\\images")
    masks = ask("Папка для BMP-масок", "data\\road\\masks")
    if images and masks: run("annotate.py", [images, masks])


def train(resume=False):
    images = ask("Папка train с BMP-кадрами", "data\\road\\train_images")
    masks = ask("Папка train с BMP-масками", "data\\road\\train_masks")
    val_images = ask("Папка validation с кадрами (Enter — случайное разбиение)", required=False)
    val_masks = ask("Папка validation с масками", required=False) if val_images else None
    if val_images and not val_masks:
        print("Для validation нужны обе папки. Операция отменена.")
        return
    epochs = ask("Количество эпох", "20" if resume else "50")
    batch = ask("Размер batch", "8")
    size = ask("Размер изображения", "192")
    output = ask("Путь к новой модели", "models\\lite_road_model_v2.pth" if resume else "models\\lite_road_model.pth")
    if not all((images, masks, epochs, batch, size, output)): return
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    args = [images, masks, "--epochs", epochs, "--batch-size", batch, "--size", size, "--output", output]
    if val_images: args += ["--val-images-dir", val_images, "--val-masks-dir", val_masks]
    if resume:
        model = ask("Путь к текущей модели")
        if not model: return
        args += ["--resume", model]
    run("train.py", args)


def infer():
    video = ask("Путь к исходному видео")
    model = ask("Путь к модели (.pth)")
    output = ask("Путь к результату MP4", "data\\result\\road_zones.mp4")
    left = ask("Левая зона (доля)", ".15")
    center = ask("Центральная зона (доля)", ".70")
    right = ask("Правая зона (доля)", ".15")
    threshold = ask("Порог дороги", ".55")
    if all((video, model, output, left, center, right, threshold)):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        run("infer.py", [video, model, "--output", output, "--left", left, "--center", center, "--right", right, "--threshold", threshold])


def menu():
    actions = {
        "1": ("Подготовить BMP-кадры из видео", prepare),
        "2": ("Разметить BMP-кадры", annotate),
        "3": ("Обучить новую модель", lambda: train(False)),
        "4": ("Дообучить существующую модель", lambda: train(True)),
        "5": ("Обработать видео", infer),
        "6": ("Проверить геометрию перекрёстков", lambda: run("-m", ["unittest", "tests/test_road_geometry.py"])),
    }
    while True:
        print("\n" + "=" * 52 + "\n  АНАЛИЗ КАРЬЕРНОЙ ДОРОГИ\n" + "=" * 52)
        for key, (title, _) in actions.items(): print(f"  {key}. {title}")
        print("  0. Выход")
        choice = input("\nВыберите действие: ").strip()
        if choice == "0": return
        action = actions.get(choice)
        if action:
            try: action[1]()
            except Exception as error: print(f"Ошибка: {error}")
        else:
            print("Неизвестный пункт меню.")


if __name__ == "__main__":
    menu()
