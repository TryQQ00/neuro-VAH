import PySimpleGUI as sg
import matplotlib.pyplot as plt
import io
import os
import threading
import queue
import logging
import time
import traceback
from typing import Dict, List, Tuple, Any, Optional, Callable

from parser import VerilogAParser
from generator import SignalGenerator
from simulator import DeviceModel
from spice_interface import SpiceSimulator
from rnn_model import PhysicsInformedRNN, ModelTrainer

import torch
import numpy as np

# Настройка логирования
logger = logging.getLogger(__name__)

# Константы для интерфейса
SUPPORTED_DEVICE_TYPES = ['Диод', 'BJT', 'HEMT', 'Memristor']
SUPPORTED_SIGNAL_TYPES = ['Синус', 'Ступень', 'Шум', 'Линейный']
SUPPORTED_MODEL_TYPES = ['Standard RNN', 'Physics-Informed RNN']
SUPPORTED_INTEGR_METHODS = ['euler', 'rk2', 'rk4', 'adaptive']

# Вспомогательная функция для отрисовки matplotlib
def draw_figure(fig, dpi=100):
    """
    Преобразует matplotlib-фигуру в изображение для PySimpleGUI.
    
    Args:
        fig (matplotlib.figure.Figure): Фигура для преобразования
        dpi (int): Разрешение изображения
        
    Returns:
        bytes: Байтовое представление изображения
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    return buf.read()

# Функция валидации ввода
def validate_input(values: Dict, key: str, 
                  type_func: Callable = float, 
                  min_val: Optional[float] = None, 
                  max_val: Optional[float] = None) -> Tuple[bool, Any]:
    """
    Проверяет корректность ввода пользователя.
    
    Args:
        values (Dict): Словарь значений из GUI
        key (str): Ключ проверяемого поля
        type_func (Callable): Функция преобразования типа
        min_val (Optional[float]): Минимальное допустимое значение
        max_val (Optional[float]): Максимальное допустимое значение
        
    Returns:
        Tuple[bool, Any]: Флаг валидности и преобразованное значение
    """
    try:
        val = type_func(values[key])
        if min_val is not None and val < min_val:
            return False, f"Значение должно быть не меньше {min_val}"
        if max_val is not None and val > max_val:
            return False, f"Значение должно быть не больше {max_val}"
        return True, val
    except (ValueError, TypeError):
        return False, f"Неверный формат данных, ожидается {type_func.__name__}"

# Функция для запуска длительных операций в отдельном потоке
def async_operation(window: sg.Window, func: Callable, *args, **kwargs) -> Any:
    """
    Запускает операцию в отдельном потоке с отображением прогресс-бара.
    
    Args:
        window (sg.Window): Окно GUI
        func (Callable): Функция для выполнения
        *args, **kwargs: Аргументы функции
        
    Returns:
        Any: Результат выполнения функции
    """
    result_queue = queue.Queue()
    progress_key = '-PROGRESS-'
    cancel_key = '-CANCEL-'
    
    # Функция-обертка для выполнения в отдельном потоке
    def thread_func():
        try:
            result = func(*args, **kwargs)
            result_queue.put(('result', result))
        except Exception as e:
            traceback.print_exc()
            result_queue.put(('error', str(e)))
    
    # Запускаем поток
    thread = threading.Thread(target=thread_func)
    thread.daemon = True
    thread.start()
    
    # Создаем окно с прогресс-баром
    layout = [
        [sg.Text('Выполняется операция, пожалуйста, подождите...')],
        [sg.ProgressBar(100, orientation='h', size=(50, 20), key=progress_key)],
        [sg.Cancel('Отмена', key=cancel_key, button_color=('white', 'red'))]
    ]
    
    progress_window = sg.Window('Прогресс', layout, modal=True, finalize=True)
    progress_bar = progress_window[progress_key]
    
    # Цикл обновления прогресс-бара
    cancelled = False
    i = 0
    while thread.is_alive():
        # Обновляем прогресс (здесь анимация, т.к. нет информации о реальном прогрессе)
        progress_bar.update(i % 100)
        i = (i + 1) % 100
        
        # Проверяем события
        event, _ = progress_window.read(timeout=100)
        if event == cancel_key or event == sg.WIN_CLOSED:
            cancelled = True
            break
    
    # Закрываем окно прогресса
    progress_window.close()
    
    # Проверяем результат
    if cancelled:
        return None
        
    if not result_queue.empty():
        status, result = result_queue.get()
        if status == 'error':
            sg.popup_error(f'Произошла ошибка:\n{result}', title='Ошибка')
            return None
        return result
    
    return None

# Создание окна GUI
def create_window():
    """
    Создает главное окно интерфейса.
    
    Returns:
        sg.Window: Объект окна
    """
    sg.theme('LightBlue3')
    
    # Верхняя панель с логом
    log_panel = [
        [sg.Text('Журнал операций:', font=('Helvetica', 10, 'bold'))],
        [sg.Multiline(size=(80, 5), key='-LOG-', autoscroll=True, disabled=True, font=('Courier', 8))]
    ]

    # Вкладка Модель
    model_tab = [
        [sg.Text('Verilog-A файл (.va):'), 
         sg.Input(key='-VAFILE-', tooltip='Путь к Verilog-A файлу модели', enable_events=True), 
         sg.FileBrowse(file_types=(('VA','*.va'),), tooltip='Выбрать файл .va')],
        [sg.Text('Тип устройства:'), 
         sg.Combo(SUPPORTED_DEVICE_TYPES, default_value='Диод', key='-DEVTYPE-', 
                  tooltip='Выберите тип прибора', readonly=True)],
        [sg.Checkbox('Использовать ngspice', key='-USE_SPICE-', 
                     tooltip='Использовать SPICE для генерации эталонных данных', enable_events=True)],
        [sg.Text('Модель:'), 
         sg.Combo(SUPPORTED_MODEL_TYPES, default_value='Standard RNN', key='-MODELTYPE-', 
                  tooltip='Стандартная RNN или гибридная модель с учётом физики', readonly=True)],
        [sg.Text('Метод интегрирования:'), 
         sg.Combo(SUPPORTED_INTEGR_METHODS, default_value='rk2', key='-INTEGRMETHOD-', 
                  tooltip='Метод численного интегрирования', readonly=True)],
        [sg.Button('Загрузить параметры', key='-LOAD_PARAMS-', 
                  tooltip='Парсить параметры из .va и показать для редактирования')],
        [sg.Frame('Параметры устройства:', [[
            sg.Column([[]], size=(400,200), scrollable=True, key='-PARAM_COL-', pad=(0,0))
        ]])]
    ]

    # Вкладка Сигнал
    signal_tab = [
        [sg.Text('Тип сигнала:'), 
         sg.Combo(SUPPORTED_SIGNAL_TYPES, default_value='Синус', key='-SIGNALTYPE-', 
                  tooltip='Форма временного входного сигнала', readonly=True)],
        [sg.Text('Длительность (сек):'), 
         sg.Input('1.0', size=(6,1), key='-DURATION-', 
                 tooltip='Длительность сигнала в секундах')],
        [sg.Text('Vmin (В):'), 
         sg.Input('0.0', size=(6,1), key='-VMIN-', 
                 tooltip='Нижняя граница напряжения (В)'), 
         sg.Text('Vmax (В):'), 
         sg.Input('1.0', size=(6,1), key='-VMAX-', 
                 tooltip='Верхняя граница напряжения (В)')],
        [sg.Text('Число точек:'), 
         sg.Input('100', size=(6,1), key='-SAMPLES-', 
                 tooltip='Количество временных точек в сигнале')],
        [sg.Frame('Дополнительные параметры:', [[
            sg.Column([
                [sg.Text('Периоды (для синуса):'), 
                 sg.Input('5', size=(6,1), key='-SINEPERIODS-', 
                         tooltip='Число периодов синусоиды')],
                [sg.Text('Фракция (для ступени):'), 
                 sg.Input('0.5', size=(6,1), key='-STEPFRAC-', 
                         tooltip='Доля времени на уровне Vmin (0-1)')]
            ], pad=(0,0))
        ]])],
        [sg.Button('Предпросмотр сигнала', key='-PREVIEW_SIGNAL-', 
                  tooltip='Показать сигнал перед генерацией')]
    ]

    # Вкладка Вариации параметров
    variation_tab = [
        [sg.Text('Настройки вариации параметров:', 
                tooltip='Генерация набора данных с разбросом параметров')],
        [sg.Text('Число наборов:'), 
         sg.Input('5', size=(6,1), key='-VAR_COUNT-', 
                 tooltip='Сколько разных параметрических наборов генерировать')],
        [sg.Text('Разброс ±:'), 
         sg.Input('0.1', size=(6,1), key='-VAR_RANGE-', 
                 tooltip='Относительный разброс параметров ± (например, 0.1)')],
        [sg.Checkbox('Использовать фиксированный seed для воспроизводимости', key='-USE_SEED-', 
                     default=True, tooltip='Обеспечить воспроизводимость результатов')],
        [sg.Text('Seed:'), 
         sg.Input('42', size=(6,1), key='-SEED-', 
                 tooltip='Значение seed для воспроизводимости')]
    ]

    # Вкладка Обучение
    train_tab = [
        [sg.Text('Параметры модели:')],
        [sg.Text('Скрытый размер:'), 
         sg.Input('64', size=(6,1), key='-HIDDEN-', 
                 tooltip='Количество нейронов в скрытом состоянии RNN'),
         sg.Text('Слои:'), 
         sg.Input('2', size=(6,1), key='-LAYERS-', 
                 tooltip='Число слоёв RNN'),
         sg.Text('Dropout:'), 
         sg.Input('0.2', size=(6,1), key='-DROPOUT-', 
                 tooltip='Вероятность dropout (0-1)')],
        [sg.Text('Параметры обучения:')],
        [sg.Text('Эпохи:'), 
         sg.Input('50', size=(6,1), key='-EPOCHS-', 
                 tooltip='Максимальное количество проходов по данным'),
         sg.Text('Batch size:'), 
         sg.Input('32', size=(6,1), key='-BATCH-', 
                 tooltip='Размер мини-батча для обучения'),
         sg.Text('Доля валидации:'), 
         sg.Input('0.2', size=(6,1), key='-VALSPLIT-', 
                 tooltip='Доля данных для валидации (0-1)')],
        [sg.Text('Learning rate:'), 
         sg.Input('0.001', size=(6,1), key='-LR-', 
                 tooltip='Скорость обучения'),
         sg.Text('Patience:'), 
         sg.Input('5', size=(6,1), key='-PATIENCE-', 
                 tooltip='Число эпох без улучшений до остановки')],
        [sg.Button('Начать обучение', key='-TRAIN-', 
                  tooltip='Запустить процесс обучения')],
        [sg.Frame('График обучения:', [[
            sg.Column([[sg.Image(key='-LOSS_IMG-')]], pad=(0,0))
        ]])]
    ]

    # Вкладка Результаты
    result_tab = [
        [sg.Button('Динамический прогноз', key='-PREDICT-', 
                  tooltip='Показать I(t) для нового сигнала'), 
         sg.Button('Построить ВАХ', key='-PLOT_IV-', 
                  tooltip='Показать статическую I–V характеристику')],
        [sg.Frame('Динамический отклик I(t):', [[
            sg.Column([[sg.Image(key='-DYN_IMG-')]], pad=(0,0))
        ]])],
        [sg.Text('Анализ динамического прогноза:', font=('Helvetica', 10, 'bold'))],
        [sg.Text('', key='-DYN_ANALYSIS-', size=(80,2), 
                tooltip='Анализ динамического прогноза')],
        [sg.Frame('Статическая ВАХ (I-V):', [[
            sg.Column([[sg.Image(key='-IV_IMG-')]], pad=(0,0))
        ]])],
        [sg.Text('Анализ статической характеристики:', font=('Helvetica', 10, 'bold'))],
        [sg.Text('', key='-IV_ANALYSIS-', size=(80,2), 
                tooltip='Анализ статической характеристики')]
    ]

    # Вкладка Метрики
    metrics_tab = [
        [sg.Button('Расчет метрик', key='-CALC_METRICS-', 
                  tooltip='Вычислить метрики качества модели')],
        [sg.Frame('Метрики качества:', [[
            sg.Column([
                [sg.Text('MSE:', font=('Helvetica', 10, 'bold')), 
                 sg.Text('', key='-MSE-', size=(15,1))],
                [sg.Text('MAE:', font=('Helvetica', 10, 'bold')), 
                 sg.Text('', key='-MAE-', size=(15,1))],
                [sg.Text('R²:', font=('Helvetica', 10, 'bold')), 
                 sg.Text('', key='-R2-', size=(15,1))]
            ], pad=(0,0))
        ]])],
        [sg.Frame('Распределение ошибок:', [[
            sg.Column([[sg.Image(key='-HIST_IMG-')]], pad=(0,0))
        ]])]
    ]

    # Вкладка Экспорт
    export_tab = [
        [sg.Text('Экспорт модели:')],
        [sg.Button('Экспорт ONNX', key='-EXPORT_ONNX-', 
                  tooltip='Экспортировать модель в формат ONNX'), 
         sg.Checkbox('Динамические оси', key='-DYNAMIC_AXES-', default=True, 
                    tooltip='Использовать динамические оси для переменной длины последовательности')],
        [sg.Text('Путь для сохранения:'), 
         sg.Input(key='-EXPORT_PATH-', size=(40,1)), 
         sg.SaveAs('Обзор', file_types=(('ONNX','*.onnx'),))],
        [sg.Text('Генерация Verilog-A:')],
        [sg.Button('Генерация Verilog-A wrapper', key='-EXPORT_VA-', 
                  tooltip='Создать Verilog-A обертку для модели')],
        [sg.Text('Путь для сохранения:'), 
         sg.Input(key='-VA_PATH-', size=(40,1)), 
         sg.SaveAs('Обзор', file_types=(('Verilog-A','*.va'),))],
        [sg.Text('Статус:', font=('Helvetica', 10, 'bold')), 
         sg.Text('', key='-EXPORT_STATUS-', size=(50,1))]
    ]

    # Объединяем все вкладки
    layout = [
        [sg.Frame('Журнал:', log_panel)],
        [sg.TabGroup([[
            sg.Tab('Модель', model_tab),
            sg.Tab('Сигнал', signal_tab),
            sg.Tab('Вариации', variation_tab),
            sg.Tab('Обучение', train_tab),
            sg.Tab('Результаты', result_tab),
            sg.Tab('Метрики', metrics_tab),
            sg.Tab('Экспорт', export_tab)
        ]], key='-TABS-')],
        [sg.Button('Выход', button_color=('white', 'firebrick'), key='Exit')]
    ]

    # Создаем окно
    window = sg.Window('RNN-Based SPICE Proxy', layout, finalize=True, resizable=True)
    
    # Настраиваем обработчик ошибок для окна
    def log_to_gui(message):
        timestamp = time.strftime("%H:%M:%S")
        window['-LOG-'].update(f"[{timestamp}] {message}\n", append=True)
        window.refresh()
        
    # Перенаправляем логи в GUI
    class GUILogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            log_to_gui(log_entry)
    
    # Настраиваем логирование
    gui_handler = GUILogHandler()
    gui_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(gui_handler)
    
    # Проверяем наличие ngspice при старте
    try:
        if SpiceSimulator.check_ngspice_installed():
            log_to_gui("✓ ngspice найден и готов к использованию")
        else:
            log_to_gui("⚠ ngspice не найден. Проверьте установку.")
            window['-USE_SPICE-'].update(False)
            window['-USE_SPICE-'].update(disabled=True)
    except Exception as e:
        log_to_gui(f"⚠ Ошибка при проверке ngspice: {e}")
        window['-USE_SPICE-'].update(False)
        window['-USE_SPICE-'].update(disabled=True)
    
    # Проверяем CUDA/MPS
    if torch.cuda.is_available():
        log_to_gui(f"✓ CUDA доступен: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        log_to_gui("✓ Apple MPS (Metal) доступен")
    else:
        log_to_gui("ℹ GPU не обнаружен, будет использован CPU")
        
    return window

# Вспомогательные функции для обработки событий GUI
def preview_signal(values: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Генерирует и отображает предварительный просмотр сигнала.
    
    Args:
        values (Dict): Словарь значений из GUI
        
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Массивы времени и напряжения или None при ошибке
    """
    try:
        # Валидация вводимых параметров
        valid, vmin = validate_input(values, '-VMIN-', float)
        if not valid:
            sg.popup_error(f"Ошибка в Vmin: {vmin}")
            return None
            
        valid, vmax = validate_input(values, '-VMAX-', float)
        if not valid:
            sg.popup_error(f"Ошибка в Vmax: {vmax}")
            return None
            
        valid, samples = validate_input(values, '-SAMPLES-', float, 10, 10000)
        if not valid:
            sg.popup_error(f"Ошибка в числе точек: {samples}")
            return None
            
        valid, duration = validate_input(values, '-DURATION-', float, 0.001)
        if not valid:
            sg.popup_error(f"Ошибка в длительности: {duration}")
            return None
        
        # Параметры зависят от типа сигнала
        signal_type = values['-SIGNALTYPE-']
        
        if signal_type == 'Синус':
            valid, periods = validate_input(values, '-SINEPERIODS-', float, 0.1)
            if not valid:
                sg.popup_error(f"Ошибка в числе периодов: {periods}")
                return None
                
            t, v = SignalGenerator.sine(vmin, vmax, int(samples), periods, duration)
            
        elif signal_type == 'Ступень':
            valid, frac = validate_input(values, '-STEPFRAC-', float, 0, 1)
            if not valid:
                sg.popup_error(f"Ошибка в значении фракции: {frac}")
                return None
                
            t, v = SignalGenerator.step(vmin, vmax, int(samples), frac, duration)
            
        elif signal_type == 'Шум':
            # Устанавливаем seed для воспроизводимости предпросмотра
            if values['-USE_SEED-']:
                valid, seed = validate_input(values, '-SEED-', int, 0)
                if not valid:
                    sg.popup_error(f"Ошибка в значении seed: {seed}")
                    return None
                SignalGenerator.set_random_seed(seed)
                
            t, v = SignalGenerator.noise(vmin, vmax, int(samples), duration)
            
        elif signal_type == 'Линейный':
            t, v = SignalGenerator.sweep(vmin, vmax, int(samples), duration)
            
        else:
            sg.popup_error(f"Неизвестный тип сигнала: {signal_type}")
            return None
        
        # Отображаем сигнал
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, v)
        ax.set_xlabel('Время (сек)')
        ax.set_ylabel('Напряжение (В)')
        ax.set_title(f'Сигнал: {signal_type}')
        ax.grid(True)
        plt.tight_layout()
        
        # Показываем график в отдельном окне
        img_bytes = draw_figure(fig)
        layout = [[sg.Image(data=img_bytes)], [sg.Button('Закрыть')]]
        preview_window = sg.Window('Предпросмотр сигнала', layout, modal=True, finalize=True)
        preview_window.read(close=True)
        
        return t, v
        
    except Exception as e:
        sg.popup_error(f"Ошибка при генерации сигнала: {str(e)}")
        return None

def generate_signal_from_values(values: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Генерирует сигнал на основе значений из GUI.
    
    Args:
        values (Dict): Словарь значений из GUI
        
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Массивы времени и напряжения или None при ошибке
    """
    try:
        # Валидация вводимых параметров
        valid, vmin = validate_input(values, '-VMIN-', float)
        if not valid:
            sg.popup_error(f"Ошибка в Vmin: {vmin}")
            return None
            
        valid, vmax = validate_input(values, '-VMAX-', float)
        if not valid:
            sg.popup_error(f"Ошибка в Vmax: {vmax}")
            return None
            
        valid, samples = validate_input(values, '-SAMPLES-', float, 10, 10000)
        if not valid:
            sg.popup_error(f"Ошибка в числе точек: {samples}")
            return None
            
        valid, duration = validate_input(values, '-DURATION-', float, 0.001)
        if not valid:
            sg.popup_error(f"Ошибка в длительности: {duration}")
            return None
        
        # Параметры зависят от типа сигнала
        signal_type = values['-SIGNALTYPE-']
        
        if signal_type == 'Синус':
            valid, periods = validate_input(values, '-SINEPERIODS-', float, 0.1)
            if not valid:
                sg.popup_error(f"Ошибка в числе периодов: {periods}")
                return None
                
            t, v = SignalGenerator.sine(vmin, vmax, int(samples), periods, duration)
            
        elif signal_type == 'Ступень':
            valid, frac = validate_input(values, '-STEPFRAC-', float, 0, 1)
            if not valid:
                sg.popup_error(f"Ошибка в значении фракции: {frac}")
                return None
                
            t, v = SignalGenerator.step(vmin, vmax, int(samples), frac, duration)
            
        elif signal_type == 'Шум':
            # Устанавливаем seed для воспроизводимости
            if values['-USE_SEED-']:
                valid, seed = validate_input(values, '-SEED-', int, 0)
                if not valid:
                    sg.popup_error(f"Ошибка в значении seed: {seed}")
                    return None
                SignalGenerator.set_random_seed(seed)
                
            t, v = SignalGenerator.noise(vmin, vmax, int(samples), duration)
            
        elif signal_type == 'Линейный':
            t, v = SignalGenerator.sweep(vmin, vmax, int(samples), duration)
            
        else:
            sg.popup_error(f"Неизвестный тип сигнала: {signal_type}")
            return None
            
        return t, v
        
    except Exception as e:
        sg.popup_error(f"Ошибка при генерации сигнала: {str(e)}")
        return None

def validate_file_exist(filepath: str) -> bool:
    """
    Проверяет существование файла.
    
    Args:
        filepath (str): Путь к файлу
        
    Returns:
        bool: True, если файл существует
    """
    if not filepath:
        sg.popup_error("Не указан путь к файлу")
        return False
        
    if not os.path.exists(filepath):
        sg.popup_error(f"Файл не найден: {filepath}")
        return False
        
    return True