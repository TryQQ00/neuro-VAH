import PySimpleGUI as sg
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import threading
import traceback
import sys
import tempfile
import base64
import json
import queue
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
from io import BytesIO
from PIL import Image

from gui import create_window, draw_figure, async_operation, validate_input, preview_signal, generate_signal_from_values, validate_file_exist
from parser import VerilogAParser
from generator import SignalGenerator
from simulator import DeviceModel
from spice_interface import SpiceSimulator
from rnn_model import PhysicsInformedRNN, ModelTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Функция для вычисления метрик качества
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Вычисляет метрики качества модели.
    
    Args:
        y_true (np.ndarray): Истинные значения
        y_pred (np.ndarray): Предсказанные значения
        
    Returns:
        Tuple[float, float, float]: MSE, MAE, R2
    """
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    var = np.var(y_true)
    r2 = 1 - np.sum((y_pred - y_true) ** 2) / (len(y_true) * var) if var > 0 else 0
    return mse, mae, r2

# Класс для хранения состояния приложения
class AppState:
    """
    Хранит состояние приложения между событиями GUI.
    """
    def __init__(self):
        self.parser = VerilogAParser()
        self.parser_info = None
        self.params = {}
        self.ports = []
        self.module = ''
        self.device_type = 'Диод'
        self.rnn_model = None
        self.trainer = None
        self.signal = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                'mps' if torch.backends.mps.is_available() else 
                                'cpu')

# Обработчики событий GUI 
class EventHandlers:
    """
    Обработчики событий GUI.
    """
    @staticmethod
    def handle_load_params(window: sg.Window, state: AppState, values: Dict) -> None:
        """
        Обрабатывает загрузку параметров из Verilog-A файла.
        
        Args:
            window (sg.Window): Окно GUI
            state (AppState): Состояние приложения
            values (Dict): Значения элементов GUI
        """
        va_file = values['-VAFILE-']
        
        if not validate_file_exist(va_file):
            return
            
        # Асинхронно парсим файл
        def parse_va():
            return state.parser.parse(va_file)
            
        parser_info = async_operation(window, parse_va)
        
        if not parser_info:
            return
            
        state.parser_info = parser_info
        state.module = parser_info['module']
        state.ports = parser_info['ports']
        
        # Обрабатываем параметры с дополнительной проверкой
        if 'params' in parser_info and parser_info['params']:
            state.params = parser_info['params'].copy()
            logger.info(f"Загружены параметры: {', '.join(state.params.keys())}")
        else:
            # Если параметры не найдены, используем базовые параметры в зависимости от типа устройства
            state.params = {}
            device_type = values['-DEVTYPE-']
            logger.warning(f"Параметры не найдены в файле {va_file}. Используем базовые параметры для {device_type}.")
            
            if device_type == 'BJT':
                state.params = {
                    "IS": 1e-15,
                    "BF": 100,
                    "BR": 1,
                    "Vt": 0.026,
                    "VAF": 100,
                    "Rth": 50.0,
                    "Cth": 1.0,
                    "alphaT": 0.005,
                    "beta_h": 0.1
                }
            elif device_type == 'Диод':
                state.params = {
                    "Is": 1e-14,
                    "N": 1.0,
                    "Rth": 50.0,
                    "Cth": 1.0
                }
            
            logger.info(f"Использованы базовые параметры: {', '.join(state.params.keys())}")
        
        # Выводим все параметры для отладки
        for k, v in state.params.items():
            logger.debug(f"Параметр: {k} = {v}")
        
        # Обновляем список параметров в GUI
        param_rows = []
        for k, v in state.params.items():
            param_rows.append([
                sg.Text(k, size=(15, 1)), 
                sg.Input(str(v), key=f'-P_{k}-', size=(15, 1))
            ])
        
        if param_rows:
            window['-PARAM_COL-'].update(param_rows)
            logger.info(f"Обновлены параметры в GUI: {len(param_rows)} параметров")
        else:
            msg = "Не найдены параметры в файле"
            logger.warning(msg)
            sg.popup(msg, title="Предупреждение")
    
    @staticmethod
    def handle_train(window: sg.Window, state: AppState, values: Dict) -> None:
        """
        Обрабатывает обучение модели.
        
        Args:
            window (sg.Window): Окно GUI
            state (AppState): Состояние приложения
            values (Dict): Значения элементов GUI
        """
        # Обновляем параметры из GUI
        if state.parser_info:
            try:
                for k in list(state.params):
                    key = f'-P_{k}-'
                    if key in values:
                        state.params[k] = float(values[key])
            except ValueError as e:
                sg.popup_error(f"Ошибка при преобразовании параметра: {e}")
                return
        
        state.device_type = values['-DEVTYPE-']
        model_type = values['-MODELTYPE-']
        
        # Валидация параметров
        valid, hidden = validate_input(values, '-HIDDEN-', int, 1, 1024)
        if not valid:
            sg.popup_error(f"Ошибка в размере скрытого слоя: {hidden}")
            return
        
        valid, layers = validate_input(values, '-LAYERS-', int, 1, 10)
        if not valid:
            sg.popup_error(f"Ошибка в количестве слоев: {layers}")
            return
        
        valid, dropout = validate_input(values, '-DROPOUT-', float, 0, 0.9)
        if not valid:
            sg.popup_error(f"Ошибка в значении dropout: {dropout}")
            return
        
        valid, epochs = validate_input(values, '-EPOCHS-', int, 1, 1000)
        if not valid:
            sg.popup_error(f"Ошибка в количестве эпох: {epochs}")
            return
        
        valid, batch_size = validate_input(values, '-BATCH-', int, 1, 1024)
        if not valid:
            sg.popup_error(f"Ошибка в размере batch: {batch_size}")
            return
        
        valid, val_split = validate_input(values, '-VALSPLIT-', float, 0.1, 0.5)
        if not valid:
            sg.popup_error(f"Ошибка в доле валидации: {val_split}")
            return
        
        valid, lr = validate_input(values, '-LR-', float, 1e-6, 1e-1)
        if not valid:
            sg.popup_error(f"Ошибка в learning rate: {lr}")
            return
        
        valid, patience = validate_input(values, '-PATIENCE-', int, 1, 100)
        if not valid:
            sg.popup_error(f"Ошибка в patience: {patience}")
            return
        
        # Генерация сигнала
        signal_result = generate_signal_from_values(values)
        if not signal_result:
            return
        t_i, v_i = signal_result
        state.signal = (t_i, v_i)  # Сохраняем для последующего использования
        
        # Параметры batch-генерации
        valid, var_count = validate_input(values, '-VAR_COUNT-', int, 1, 100)
        if not valid:
            sg.popup_error(f"Ошибка в количестве вариаций: {var_count}")
            return
            
        valid, var_frac = validate_input(values, '-VAR_RANGE-', float, 0, 1)
        if not valid:
            sg.popup_error(f"Ошибка в разбросе параметров: {var_frac}")
            return
            
        # Seed для воспроизводимости
        seed = None
        if values['-USE_SEED-']:
            valid, seed_val = validate_input(values, '-SEED-', int, 0)
            if not valid:
                sg.popup_error(f"Ошибка в значении seed: {seed_val}")
                return
            seed = seed_val
        
        # Асинхронно обучаем модель
        def train_model():
            # Настраиваем генерацию сигналов
            signal_map = {
                'Синус': SignalGenerator.sine,
                'Ступень': SignalGenerator.step,
                'Шум': SignalGenerator.noise,
                'Линейный': SignalGenerator.sweep
            }
            
            # Параметры генерации
            signal_params = {
                'vmin': float(values['-VMIN-']),
                'vmax': float(values['-VMAX-']),
                'samples': int(float(values['-SAMPLES-'])),
                'duration': float(values['-DURATION-'])
            }
            
            # Дополнительные параметры в зависимости от типа сигнала
            signal_type = values['-SIGNALTYPE-']
            if signal_type == 'Синус':
                signal_params['periods'] = float(values['-SINEPERIODS-'])
            elif signal_type == 'Ступень':
                signal_params['frac'] = float(values['-STEPFRAC-'])
            
            gen_func = lambda **kwargs: signal_map[signal_type](**kwargs)
            
            # Генерация набора данных с вариациями
            if seed is not None:
                SignalGenerator.set_random_seed(seed)
            
            # Фильтруем параметры устройства от параметров сигнала
            device_params = state.params.copy()
            
            # Список стандартных параметров сигнала
            signal_param_names = {'vmin', 'vmax', 'samples', 'duration', 'periods', 'frac'}
            
            # Генерируем один сигнал без вариаций сначала
            t_base, v_base = gen_func(**signal_params)
            
            # Формирование обучающих данных - будем создавать вариации вручную
            V_list, Y_list = [], []
            for i in range(var_count):
                # Создаем вариацию параметров устройства для этого сигнала
                p_i = {
                    k: v * (1 + np.random.uniform(-var_frac, var_frac))
                    for k, v in device_params.items()
                }
                
                # Выбор физической модели или SPICE
                if values['-USE_SPICE-']:
                    _, Iphys = SpiceSimulator.run(
                        values['-VAFILE-'], 
                        state.module, 
                        state.ports, 
                        p_i, 
                        v_base,
                        dt=float(values['-DURATION-']) / len(v_base) if len(v_base) > 0 else None
                    )
                else:
                    # Используем указанный метод интегрирования
                    integr_method = values['-INTEGRMETHOD-']
                    Iphys = DeviceModel(p_i, state.device_type).simulate(
                        v_base, 
                        dt=float(values['-DURATION-']) / len(v_base) if len(v_base) > 0 else None,
                        method=integr_method
                    )
                
                V_list.append(v_base)
                Y_list.append(Iphys)
            
            # Преобразование в тензоры
            V_arr = np.stack(V_list)
            Y_arr = np.stack(Y_list)
            Vt = torch.tensor(V_arr[..., None], dtype=torch.float32)
            Yt = torch.tensor(Y_arr[..., None], dtype=torch.float32)
            
            # Создание модели
            state.rnn_model = PhysicsInformedRNN(hidden, layers, dropout)
            
            # Функция физики
            if model_type == 'Standard RNN':
                phys_func = lambda X: np.zeros_like(X)
            else:
                def phys_func(X):
                    # Проверяем формат входных данных
                    if isinstance(X, np.ndarray):
                        if X.ndim == 1:  # одна последовательность [seq_len]
                            result = DeviceModel(state.params, state.device_type).simulate(
                                X,
                                dt=float(values['-DURATION-']) / (len(X) if len(X) > 0 else 1),
                                method=values['-INTEGRMETHOD-']
                            )
                            return result
                        elif X.ndim == 2:  # батч последовательностей [batch, seq_len]
                            # Обрабатываем каждую последовательность отдельно
                            results = []
                            for x in X:
                                i = DeviceModel(state.params, state.device_type).simulate(
                                    x,
                                    dt=float(values['-DURATION-']) / (len(x) if len(x) > 0 else 1),
                                    method=values['-INTEGRMETHOD-']
                                )
                                results.append(i)
                            return np.array(results)
                        else:
                            # Неожиданная размерность, возвращаем нули
                            logger.warning(f"Неожиданная размерность входных данных в phys_func: {X.shape}")
                            return np.zeros_like(X)
                    else:
                        # Скаляр или другой формат
                        try:
                            x_arr = np.array([X] if np.isscalar(X) else X)
                            result = DeviceModel(state.params, state.device_type).simulate(
                                x_arr,
                                dt=float(values['-DURATION-']),
                                method=values['-INTEGRMETHOD-']
                            )
                            return result
                        except Exception as e:
                            logger.error(f"Ошибка при обработке входных данных в phys_func: {e}")
                            return 0.0
            
            # Создание тренера
            state.trainer = ModelTrainer(
                state.rnn_model, 
                state.device, 
                phys_func,
                learning_rate=lr,
                weight_decay=1e-5,
                checkpoint_dir='checkpoints'
            )
            
            # Запуск обучения
            history = state.trainer.train(
                Vt, 
                Yt,
                epochs=epochs,
                batch_size=batch_size,
                val_split=val_split,
                patience=patience,
                seed=seed
            )
            
            return history
        
        # Запускаем обучение асинхронно и получаем результаты
        history = async_operation(window, train_model)
        
        if not history:
            return
            
        # Визуализация графика обучения
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history['train_losses'], label='Тренировка')
        ax.plot(history['val_losses'], label='Валидация')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('MSE')
        ax.set_title('Процесс обучения')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        
        # Обновляем график в GUI
        window['-LOSS_IMG-'].update(data=draw_figure(fig))
    
    @staticmethod
    def handle_predict(window: sg.Window, state: AppState, values: Dict) -> None:
        """
        Обрабатывает предсказание динамического отклика.
        
        Args:
            window (sg.Window): Окно GUI
            state (AppState): Состояние приложения
            values (Dict): Значения элементов GUI
        """
        if state.trainer is None:
            sg.popup_error('Сначала обучите модель')
            return
        
        # Генерация нового сигнала, если не сохранен существующий
        if state.signal is None:
            signal_result = generate_signal_from_values(values)
            if not signal_result:
                return
            t, v = signal_result
            state.signal = (t, v)
        else:
            t, v = state.signal
        
        # Асинхронно выполняем предсказание
        def predict():
            # Выбор физической модели или SPICE
            if values['-USE_SPICE-']:
                _, Iphys = SpiceSimulator.run(
                    values['-VAFILE-'], 
                    state.module, 
                    state.ports, 
                    state.params, 
                    v,
                    dt=float(values['-DURATION-']) / len(v) if len(v) > 0 else None
                )
            else:
                # Используем указанный метод интегрирования
                integr_method = values['-INTEGRMETHOD-']
                Iphys = DeviceModel(state.params, state.device_type).simulate(
                    v,
                    dt=float(values['-DURATION-']) / len(v) if len(v) > 0 else None,
                    method=integr_method
                )
            
            try:
                # Преобразуем данные в тензоры
                Vr = torch.tensor(v[None, ..., None], dtype=torch.float32).to(state.device)
                Ip = torch.tensor(Iphys[None, ..., None], dtype=torch.float32).to(state.device)
                
                # Проверяем размерности и логируем их для отладки
                logger.debug(f"Размерность Vr: {Vr.shape}, Ip: {Ip.shape}")
                
                # Выполняем прогноз
                state.rnn_model.eval()
                with torch.no_grad():
                    try:
                        Ypred = state.rnn_model(Vr, Ip).cpu().numpy().reshape(-1)
                    except RuntimeError as e:
                        # Попробуем исправить размерности, если возникла ошибка
                        logger.error(f"Ошибка при прогнозе: {e}")
                        logger.info("Пробуем исправить размерности тензоров...")
                        
                        # Попробуем другие форматы тензоров
                        if len(v.shape) == 1:
                            # Пробуем переформатировать как [batch=1, seq_len, features=1]
                            Vr = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(state.device)
                            Ip = torch.tensor(Iphys, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(state.device)
                            Ypred = state.rnn_model(Vr, Ip).cpu().numpy().reshape(-1)
                
                return t, Iphys, Ypred
            except Exception as e:
                logger.error(f"Ошибка при расчете динамического отклика: {e}")
                return t, Iphys, np.zeros_like(Iphys)
        
        # Выполняем предсказание и получаем результаты
        result = async_operation(window, predict)
        
        if not result:
            return
            
        t, Iphys, Ypred = result
        
        # Построение графика
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, Iphys, label='Физическая модель')
        ax.plot(t, Ypred, '--', label='RNN')
        ax.set_xlabel('Время (сек)')
        ax.set_ylabel('Ток (А)')
        ax.set_title('Динамический отклик I(t)')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        
        # Обновляем график в GUI
        window['-DYN_IMG-'].update(data=draw_figure(fig))
        
        # Анализ качества
        err = Ypred - Iphys
        mse, mae, r2 = compute_metrics(Iphys, Ypred)
        
        analysis = (
            f"Максимальное отклонение: {np.max(np.abs(err)):.3e} | "
            f"Средняя ошибка: {np.mean(err):.3e} | "
            f"MSE: {mse:.3e} | MAE: {mae:.3e} | R²: {r2:.3f}"
        )
        window['-DYN_ANALYSIS-'].update(analysis)
    
    @staticmethod
    def handle_plot_iv(window: sg.Window, state: AppState, values: Dict) -> None:
        """
        Обрабатывает построение статической ВАХ.
        
        Args:
            window (sg.Window): Окно GUI
            state (AppState): Состояние приложения
            values (Dict): Значения элементов GUI
        """
        if not hasattr(state, 'rnn_model') or state.rnn_model is None:
            sg.popup_error("Сначала нужно обучить модель!")
            return
            
        # Валидация параметров
        valid, vmin = validate_input(values, '-VMIN-', float, -100, 100)
        if not valid:
            sg.popup_error(f"Ошибка в значении Vmin: {vmin}")
            return
            
        valid, vmax = validate_input(values, '-VMAX-', float, -100, 100)
        if not valid:
            sg.popup_error(f"Ошибка в значении Vmax: {vmax}")
            return
            
        valid, samples = validate_input(values, '-IVSAMPLES-', int, 5, 1000)
        if not valid:
            sg.popup_error(f"Ошибка в количестве точек: {samples}")
            return
        
        # Асинхронно рассчитываем ВАХ
        def calculate_iv():
            try:
                # Создаем линейную развертку напряжения
                logger.debug(f"Создаем линейную развертку напряжения: от {vmin}В до {vmax}В, {samples} точек")
                _, v_swp = SignalGenerator.sweep(vmin, vmax, int(samples))
                
                # Рассчитываем физический ток
                if values['-USE_SPICE-']:
                    logger.debug("Используем SPICE для расчета тока")
                    _, Ip = SpiceSimulator.run(
                        values['-VAFILE-'], 
                        state.module, 
                        state.ports, 
                        state.params, 
                        v_swp
                    )
                else:
                    logger.debug(f"Используем DeviceModel для расчета тока, тип устройства: {state.device_type}")
                    # Для статической ВАХ используем метод simulate_static_iv
                    try:
                        _, Ip = DeviceModel(state.params, state.device_type).simulate_static_iv(vmin, vmax, int(samples))
                        logger.debug(f"Рассчитаны токи с помощью simulate_static_iv, диапазон значений: {np.min(Ip):.3e} - {np.max(Ip):.3e}")
                    except AttributeError:
                        # Если метод не поддерживается, используем обычную симуляцию
                        logger.warning("Метод simulate_static_iv не поддерживается, используем обычную симуляцию")
                        Ip = DeviceModel(state.params, state.device_type).simulate(v_swp)
                    except Exception as e:
                        logger.error(f"Ошибка при расчете ВАХ: {e}")
                        # Используем обычную симуляцию как запасной вариант
                        logger.warning("Используем обычную симуляцию в качестве запасного варианта")
                        try:
                            Ip = DeviceModel(state.params, state.device_type).simulate(v_swp)
                        except Exception as e2:
                            logger.error(f"Ошибка при запасном расчете ВАХ: {e2}")
                            return None
                
                try:
                    # Преобразуем данные в тензоры для модели
                    logger.debug(f"Преобразуем данные в тензоры: v_swp shape = {v_swp.shape}, Ip shape = {Ip.shape}")
                    Vr = torch.tensor(v_swp[None, ..., None], dtype=torch.float32).to(state.device)
                    Ip_t = torch.tensor(Ip[None, ..., None], dtype=torch.float32).to(state.device)
                    
                    # Проверяем размерности и логируем их для отладки
                    logger.debug(f"Размерность Vr: {Vr.shape}, Ip_t: {Ip_t.shape}")
                    
                    # Выполняем прогноз
                    state.rnn_model.eval()
                    with torch.no_grad():
                        try:
                            logger.debug("Запускаем прогноз с помощью rnn_model")
                            Yp_iv = state.rnn_model(Vr, Ip_t).cpu().numpy().reshape(-1)
                            logger.debug(f"Прогноз успешно выполнен, shape = {Yp_iv.shape}")
                        except RuntimeError as e:
                            # Попробуем исправить размерности, если возникла ошибка
                            logger.error(f"Ошибка при прогнозе: {e}")
                            logger.info("Пробуем исправить размерности тензоров...")
                            
                            # Попробуем другие форматы тензоров
                            if len(v_swp.shape) == 1:
                                logger.debug("Пробуем переформатировать как [batch=1, seq_len, features=1]")
                                # Пробуем переформатировать как [batch=1, seq_len, features=1]
                                Vr = torch.tensor(v_swp, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(state.device)
                                Ip_t = torch.tensor(Ip, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(state.device)
                                Yp_iv = state.rnn_model(Vr, Ip_t).cpu().numpy().reshape(-1)
                                logger.debug(f"Прогноз успешно выполнен после переформатирования, shape = {Yp_iv.shape}")
                    
                    return v_swp, Ip, Yp_iv
                except Exception as e:
                    logger.error(f"Ошибка при расчете ВАХ: {e}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
                    return None
            except Exception as e:
                logger.error(f"Неожиданная ошибка при расчете ВАХ: {e}")
                logger.error(f"Трассировка: {traceback.format_exc()}")
                return None
        
        # Выполняем расчет и получаем результаты
        result = async_operation(window, calculate_iv)
        
        if not result:
            sg.popup_error("Не удалось рассчитать ВАХ. Проверьте логи для подробностей.")
            return
            
        v_swp, Ip, Yp_iv = result
        
        # Построение графика
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(v_swp, Ip, '-o', label='Физическая модель')
        ax.plot(v_swp, Yp_iv, '--x', label='RNN')
        ax.set_xlabel('Напряжение (В)')
        ax.set_ylabel('Ток (А)')
        ax.set_title('Статическая ВАХ (I–V)')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        
        # Обновляем график в GUI
        window['-IV_IMG-'].update(data=draw_figure(fig))
        
        # Анализ качества
        mse, mae, r2 = compute_metrics(Ip, Yp_iv)
        
        analysis = (
            f"MSE: {mse:.3e} | "
            f"MAE: {mae:.3e} | "
            f"R²: {r2:.3f}"
        )
        window['-IV_ANALYSIS-'].update(analysis)
    
    @staticmethod
    def handle_calc_metrics(window: sg.Window, state: AppState, values: Dict) -> None:
        """
        Обрабатывает расчет метрик модели.
        
        Args:
            window (sg.Window): Окно GUI
            state (AppState): Состояние приложения
            values (Dict): Значения элементов GUI
        """
        if not hasattr(state, 'rnn_model') or state.rnn_model is None:
            sg.popup_error("Сначала нужно обучить модель!")
            return
            
        # Валидация параметров
        valid, vmin = validate_input(values, '-VMIN-', float, -100, 100)
        if not valid:
            sg.popup_error(f"Ошибка в значении Vmin: {vmin}")
            return
            
        valid, vmax = validate_input(values, '-VMAX-', float, -100, 100)
        if not valid:
            sg.popup_error(f"Ошибка в значении Vmax: {vmax}")
            return
            
        valid, samples = validate_input(values, '-IVSAMPLES-', int, 5, 1000)
        if not valid:
            sg.popup_error(f"Ошибка в количестве точек: {samples}")
            return
        
        # Асинхронно рассчитываем метрики
        def calculate_metrics():
            try:
                # Создаем линейную развертку напряжения
                logger.debug(f"Создаем линейную развертку напряжения: от {vmin}В до {vmax}В, {samples} точек")
                _, v_swp = SignalGenerator.sweep(vmin, vmax, int(samples))
                
                # Рассчитываем физический ток
                if values['-USE_SPICE-']:
                    logger.debug("Используем SPICE для расчета тока")
                    _, Ip = SpiceSimulator.run(
                        values['-VAFILE-'], 
                        state.module, 
                        state.ports, 
                        state.params, 
                        v_swp
                    )
                else:
                    logger.debug(f"Используем DeviceModel для расчета тока, тип устройства: {state.device_type}")
                    # Для статической ВАХ используем метод simulate_static_iv
                    try:
                        _, Ip = DeviceModel(state.params, state.device_type).simulate_static_iv(vmin, vmax, int(samples))
                        logger.debug(f"Рассчитаны токи с помощью simulate_static_iv, диапазон значений: {np.min(Ip):.3e} - {np.max(Ip):.3e}")
                    except AttributeError:
                        # Если метод не поддерживается, используем обычную симуляцию
                        logger.warning("Метод simulate_static_iv не поддерживается, используем обычную симуляцию")
                        Ip = DeviceModel(state.params, state.device_type).simulate(v_swp)
                    except Exception as e:
                        logger.error(f"Ошибка при расчете ВАХ: {e}")
                        # Используем обычную симуляцию как запасной вариант
                        logger.warning("Используем обычную симуляцию в качестве запасного варианта")
                        try:
                            Ip = DeviceModel(state.params, state.device_type).simulate(v_swp)
                        except Exception as e2:
                            logger.error(f"Ошибка при запасном расчете ВАХ: {e2}")
                            return None
                
                try:
                    # Преобразуем данные в тензоры для модели
                    logger.debug(f"Преобразуем данные в тензоры: v_swp shape = {v_swp.shape}, Ip shape = {Ip.shape}")
                    Vr = torch.tensor(v_swp[None, ..., None], dtype=torch.float32).to(state.device)
                    Ip_t = torch.tensor(Ip[None, ..., None], dtype=torch.float32).to(state.device)
                    
                    # Проверяем размерности и логируем их для отладки
                    logger.debug(f"Размерность Vr: {Vr.shape}, Ip_t: {Ip_t.shape}")
                    
                    # Выполняем прогноз
                    state.rnn_model.eval()
                    with torch.no_grad():
                        try:
                            logger.debug("Запускаем прогноз с помощью rnn_model")
                            Yp_iv = state.rnn_model(Vr, Ip_t).cpu().numpy().reshape(-1)
                            logger.debug(f"Прогноз успешно выполнен, shape = {Yp_iv.shape}")
                        except RuntimeError as e:
                            # Попробуем исправить размерности, если возникла ошибка
                            logger.error(f"Ошибка при прогнозе: {e}")
                            logger.info("Пробуем исправить размерности тензоров...")
                            
                            # Попробуем другие форматы тензоров
                            if len(v_swp.shape) == 1:
                                logger.debug("Пробуем переформатировать как [batch=1, seq_len, features=1]")
                                # Пробуем переформатировать как [batch=1, seq_len, features=1]
                                Vr = torch.tensor(v_swp, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(state.device)
                                Ip_t = torch.tensor(Ip, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(state.device)
                                Yp_iv = state.rnn_model(Vr, Ip_t).cpu().numpy().reshape(-1)
                                logger.debug(f"Прогноз успешно выполнен после переформатирования, shape = {Yp_iv.shape}")
                    
                    # Рассчитываем метрики
                    logger.debug("Рассчитываем метрики качества")
                    mse, mae, r2 = compute_metrics(Ip, Yp_iv)
                    logger.debug(f"Рассчитаны метрики: MSE = {mse:.3e}, MAE = {mae:.3e}, R² = {r2:.3f}")
                    
                    # Рассчитываем ошибки для гистограммы
                    errors = Yp_iv - Ip
                    
                    return mse, mae, r2, errors
                except Exception as e:
                    logger.error(f"Ошибка при расчете метрик: {e}")
                    logger.error(f"Трассировка: {traceback.format_exc()}")
                    return None
            except Exception as e:
                logger.error(f"Неожиданная ошибка при расчете метрик: {e}")
                logger.error(f"Трассировка: {traceback.format_exc()}")
                return None
        
        # Выполняем расчет и получаем результаты
        result = async_operation(window, calculate_metrics)
        
        if not result:
            sg.popup_error("Не удалось рассчитать метрики. Проверьте логи для подробностей.")
            return
            
        mse, mae, r2, errors = result
        
        # Обновляем метрики в GUI
        window['-MSE-'].update(f"{mse:.3e}")
        window['-MAE-'].update(f"{mae:.3e}")
        window['-R2-'].update(f"{r2:.3f}")
        
        # Построение гистограммы ошибок
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(errors, bins=30)
        ax.set_xlabel('Ошибка')
        ax.set_ylabel('Количество')
        ax.set_title('Распределение ошибок модели')
        ax.grid(True)
        plt.tight_layout()
        
        # Обновляем гистограмму в GUI
        window['-HIST_IMG-'].update(data=draw_figure(fig))
    
    @staticmethod
    def handle_export_onnx(window: sg.Window, state: AppState, values: Dict) -> None:
        """
        Обрабатывает экспорт модели в формат ONNX.
        
        Args:
            window (sg.Window): Окно GUI
            state (AppState): Состояние приложения
            values (Dict): Значения элементов GUI
        """
        if state.trainer is None:
            sg.popup_error('Сначала обучите модель')
            return
            
        # Проверяем путь сохранения
        path = values['-EXPORT_PATH-']
        if not path:
            path = sg.popup_get_file(
                'Сохранить ONNX-модель', 
                save_as=True, 
                default_extension='.onnx',
                file_types=(('ONNX','*.onnx'),)
            )
            if not path:
                return
                
        # Проверяем директорию сохранения
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        except Exception as e:
            sg.popup_error(f"Не удалось создать директорию: {e}")
            return
            
        # Асинхронно выполняем экспорт
        def export_onnx():
            # Получаем параметры экспорта
            use_dynamic_axes = values['-DYNAMIC_AXES-']
            
            # Валидация параметров
            valid, samples = validate_input(values, '-SAMPLES-', int, 10, 10000)
            if not valid:
                raise ValueError(f"Ошибка в числе точек: {samples}")
                
            # Экспортируем модель
            state.trainer.save_onnx(path, seq_len=samples, dynamic_axes=use_dynamic_axes)
            
            return path
            
        # Выполняем экспорт и получаем результат
        result = async_operation(window, export_onnx)
        
        if not result:
            return
            
        # Обновляем статус
        window['-EXPORT_STATUS-'].update(f"ONNX-модель сохранена в {result}")
    
    @staticmethod
    def handle_export_va(window: sg.Window, state: AppState, values: Dict) -> None:
        """
        Обрабатывает генерацию Verilog-A wrapper.
        
        Args:
            window (sg.Window): Окно GUI
            state (AppState): Состояние приложения
            values (Dict): Значения элементов GUI
        """
        if not state.parser_info:
            sg.popup_error('Сначала загрузите Verilog-A файл')
            return
            
        # Проверяем путь сохранения
        path = values['-VA_PATH-']
        if not path:
            path = sg.popup_get_file(
                'Сохранить Verilog-A wrapper', 
                save_as=True, 
                default_extension='.va',
                file_types=(('Verilog-A','*.va'),)
            )
            if not path:
                return
                
        # Проверяем директорию сохранения
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        except Exception as e:
            sg.popup_error(f"Не удалось создать директорию: {e}")
            return
            
        # Генерируем Verilog-A wrapper
        try:
            with open(path, 'w') as f:
                f.write('`include "disciplines.vams"\n\n')
                
                # Имя модуля с постфиксом _rnn
                module_name = f"{state.module}_rnn"
                
                # Список портов
                ports_str = ', '.join(state.ports)
                
                # Заголовок модуля
                f.write(f'module {module_name}({ports_str});\n')
                
                # Объявление портов
                port_directions = state.parser_info.get('port_directions', {})
                for port in state.ports:
                    direction = port_directions.get(port, 'inout')
                    f.write(f'  {direction} {port};\n')
                
                # Тепловые параметры, если есть
                if any(p in state.params for p in ['Rth', 'Cth', 'alphaT', 'tauH']):
                    f.write('\n  // Тепловые параметры\n')
                    if 'Rth' in state.params:
                        f.write(f'  parameter real Rth = {state.params["Rth"]};\n')
                    if 'Cth' in state.params:
                        f.write(f'  parameter real Cth = {state.params["Cth"]};\n')
                    if 'alphaT' in state.params:
                        f.write(f'  parameter real alphaT = {state.params["alphaT"]};\n')
                
                # Внешняя переменная для вызова RNN
                f.write('\n  // Внешние переменные\n')
                if len(state.ports) == 2:
                    # Для двухпортового устройства (диод, мемристор)
                    f.write(f'  voltage v, i;\n')
                    f.write(f'  v = V({state.ports[0]}, {state.ports[1]});\n')
                    f.write(f'  I({state.ports[0]}, {state.ports[1]}) <+ i;\n')
                elif len(state.ports) == 3:
                    # Для трехпортового устройства (транзистор)
                    f.write(f'  voltage vgs, vds, i;\n')
                    f.write(f'  vgs = V({state.ports[1]}, {state.ports[2]});\n')
                    f.write(f'  vds = V({state.ports[0]}, {state.ports[2]});\n')
                    f.write(f'  I({state.ports[0]}, {state.ports[2]}) <+ i;\n')
                
                # Описание модуля
                f.write('\n  // Здесь будет вызов ONNX Runtime для расчёта I(t)\n')
                f.write('  // Пример псевдокода:\n')
                f.write('  // i = $onnx_eval("model.onnx", v);\n')
                
                # Закрытие модуля
                f.write('endmodule\n')
        except Exception as e:
            sg.popup_error(f"Ошибка при генерации Verilog-A wrapper: {e}")
            return
            
        # Обновляем статус
        window['-EXPORT_STATUS-'].update(f"Verilog-A wrapper сохранен в {path}")
        
    @staticmethod
    def handle_preview_signal(window: sg.Window, state: AppState, values: Dict) -> None:
        """
        Обрабатывает предпросмотр сигнала.
        
        Args:
            window (sg.Window): Окно GUI
            state (AppState): Состояние приложения
            values (Dict): Значения элементов GUI
        """
        preview_signal(values)

def main():
    """
    Главная функция приложения.
    """
    try:
        # Создание окна и инициализация переменных
        window = create_window()
        state = AppState()
        
        logger.info(f'Используется устройство: {state.device}')
        
        # Главный цикл обработки событий
        while True:
            event, values = window.read()
            
            # Выход из приложения
            if event in (sg.WIN_CLOSED, 'Exit'):
                break
                
            try:
                # Диспетчеризация событий
                if event == '-LOAD_PARAMS-':
                    EventHandlers.handle_load_params(window, state, values)
                    
                elif event == '-TRAIN-':
                    EventHandlers.handle_train(window, state, values)
                    
                elif event == '-PREDICT-':
                    EventHandlers.handle_predict(window, state, values)
                    
                elif event == '-PLOT_IV-':
                    EventHandlers.handle_plot_iv(window, state, values)
                    
                elif event == '-CALC_METRICS-':
                    EventHandlers.handle_calc_metrics(window, state, values)
                    
                elif event == '-EXPORT_ONNX-':
                    EventHandlers.handle_export_onnx(window, state, values)
                    
                elif event == '-EXPORT_VA-':
                    EventHandlers.handle_export_va(window, state, values)
                    
                elif event == '-PREVIEW_SIGNAL-':
                    EventHandlers.handle_preview_signal(window, state, values)
                    
            except Exception as e:
                # Обработка исключений
                logger.error(f"Ошибка при обработке события {event}: {e}")
                logger.error(traceback.format_exc())
                sg.popup_error(f"Произошла ошибка: {e}")
        
        # Закрытие окна
        window.close()
        
    except Exception as e:
        # Обработка критических исключений
        logger.critical(f"Критическая ошибка: {e}")
        logger.critical(traceback.format_exc())
        sg.popup_error(f"Критическая ошибка: {e}\n\nПодробности см. в логах.")

if __name__ == '__main__':
    main()