# Быстрая сегментация карьерной дороги

Проект заменяет тяжёлую U-Net на `LiteRoadNet` с depthwise-separable свёртками. Она обучается на маске дороги, а после сегментации сохраняется только один связный компонент, подходящий к нижней части кадра. Поэтому отдельные пятна на откосах, небе и технике не становятся «дорогой».

Старая модель `best_road_model.pth` несовместима с новой архитектурой: сначала обучите новую.

## Полный порядок работы

Ниже показаны команды для PowerShell. Перед началом активируйте виртуальное окружение, подставив свой путь:

```powershell
# один раз: создать и активировать окружение
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# установить зависимости
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 1. Подготовка обучающих данных

Извлеките кадры из видео. `--every 10` сохраняет каждый десятый кадр; для разнообразных сцен лучше брать кадры из разных рейсов, погоды и времени суток.

```powershell
python prepare.py C:\data\raw_video.mp4 C:\data\road\images --every 10
```

Разметьте каждый кадр в любом редакторе масок: дорога — белым, всё остальное — чёрным. Маска должна лежать в отдельной папке и иметь то же имя с суффиксом `_mask`:

```text
C:\data\road\images\frame_000000.bmp
C:\data\road\masks\frame_000000_mask.bmp
```

Подойдут BMP-маски того же размера, что и кадры. Полезно размечать именно участки, где модель раньше ошибалась: откосы, кучи породы, техника, тени и другие дороги.

### 2. Обучение с нуля

```powershell
python train.py C:\data\road\images C:\data\road\masks --epochs 50 --batch-size 16 --size 192 --output C:\data\models\lite_road_model.pth
```

Если на устройстве не хватает памяти, уменьшите `--batch-size` до `8` или `4`. `--size 192` — стартовый вариант для CPU; для более точной, но медленной модели можно выбрать `256`.

### 3. Дообучение на новых ошибочных кадрах

Соберите и разметьте новые кадры, затем продолжите обучение. Разрешение `--size` должно быть таким же, как у исходной модели.

```powershell
python train.py C:\data\new_images C:\data\new_masks --epochs 20 --batch-size 8 --size 192 --resume C:\data\models\lite_road_model.pth --output C:\data\models\lite_road_model_v2.pth
```

### 4. Обработка видео

```powershell
python infer.py C:\data\test_video.mp4 C:\data\models\lite_road_model.pth --output C:\data\result.mp4 --left 0.15 --center 0.70 --right 0.15
```

Чтобы посмотреть обработку без сохранения файла, уберите `--output`. Для запуска без окна просмотра (например, на сервере) добавьте `--no-display`:

```powershell
python infer.py C:\data\test_video.mp4 C:\data\models\lite_road_model.pth --no-display --output C:\data\result.mp4
```

### 5. Настройка зон и чувствительности

Доли зон обязаны суммироваться в `1.0`. Левая и правая зоны рисуются жёлтым, центральная — зелёным, всё вне подтверждённой дороги — красным. Нажмите `q`, чтобы остановить показ видео.

```powershell
# 20% слева, 60% центр, 20% справа; меньше ложных срабатываний
python infer.py C:\data\test_video.mp4 C:\data\models\lite_road_model.pth --left 0.20 --center 0.60 --right 0.20 --threshold 0.65

# Более чувствительное обнаружение (может увеличить ложные срабатывания)
python infer.py C:\data\test_video.mp4 C:\data\models\lite_road_model.pth --threshold 0.45
```

Для CPU начните с разрешения обучения 192×192. Если точность недостаточна, увеличьте `--size` при обучении до 256; если важнее скорость — используйте 160. Реальный FPS зависит от процессора, разрешения кадра и установленного PyTorch; замер выводится на видео.
