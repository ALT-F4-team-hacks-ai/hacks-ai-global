<h1 align="center">Международный хакатон Цифровой прорыв, кейс: <b>Интеллектуальный ассистент методиста</b></h1>

### Основные файлы и папки:

<ul>
<li><b><i>notebooks</i></b> - папка с ноутбуками обучения и инференса моделей</li>
<li><b><i>models</i></b> - папка с моделями</li>
<li><b><i>pipeline.py</i></b> - файл с итоговым пайплайном</li>
</ul>

### Инструкция по запуску инференса:
<span><i>на вход идёт аудиофайл на выход - все термины, содержащиеся в этом аудиофайле</i></span>

<span>!!! Для корректной транскрибации на CPU необходимо установить `conda`. </span>
<span>После установки выполнить команду `conda install -c conda-forge ffmpeg`</span>

1. Перейти в папку <i>notebooks/inference</i>
2. Команда `pip install -r requirements.txt`
3. Положить ваш аудиофайл в папку <i>data</i>
4. Вставить имя файла <a href="#">сюда</a>
5. Запустить инференс модули - `python inference.py`
6. Получить результат в качестве .csv файла в папке <i>result</i>
