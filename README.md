<h1 align="center">Международный хакатон Цифровой прорыв, кейс: <b>Интеллектуальный ассистент методиста</b></h1>

### Основные файлы и папки:

<ul>
<li><b><i>notebooks</i></b> - папка с ноутбуками обучения и прочими</li>
<li><b><i>models</i></b> - папка с моделями</li>
<li><b><i>pipeline.py</i></b> - файл с итоговым пайплайном</li>
</ul>

### Инструкция по запуску инференса:
<span><i>на вход идёт аудиофайл на выход - все термины, содержащиеся в этом аудиофайле</i></span>

<span>!!! Для корректной транскрибации на CPU необходимо установить `conda`. </span>
<span>После установки выполнить команду `conda install -c conda-forge ffmpeg`</span>

1. Создать папку models в корне
2. зайти в неё и выполнить команду `gdown 1OeG5kHSCIWPOWFcgW6_2zIyR2Gc6yn3k`
3. Перейти в папку <i>inference</i>
4. Команда `pip install -r requirements.txt`
5. Положить ваш аудиофайл в папку <i>data</i>
6. Вставить имя файла <a href="https://github.com/ALT-F4-team-hacks-ai/hacks-ai-global/blob/e54f621c3e4545341239ab3ea41f3e1f8e29f209/inference/inference.py#L294">сюда</a>
7. Запустить инференс модули - `python inference.py`
8. Получить результат в качестве .csv файла в папке <i>result</i>

### Инструкция по запуску веб-сайта:

<span>!!! Для корректной транскрибации на CPU необходимо установить `conda`. </span>
<span>После установки выполнить команду `conda install -c conda-forge ffmpeg`</span>

1. Команда `pip install -r requirements.txt`
2. Команда `python main.py`
