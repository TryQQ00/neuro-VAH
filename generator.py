import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Генерация временных и статических сигналов, а также batch-генерация вариаций.
    
    Методы класса:
    - sine: Генерация синусоидального сигнала
    - step: Генерация ступенчатого сигнала
    - noise: Генерация случайного шума
    - sweep: Генерация линейно возрастающего сигнала
    - batch_variation: Пакетная генерация сигналов с вариациями параметров
    - set_random_seed: Установка seed для воспроизводимости результатов
    """
    
    @staticmethod
    def set_random_seed(seed: int = 42) -> None:
        """
        Устанавливает seed для генератора случайных чисел NumPy.
        
        Args:
            seed (int): Значение seed для инициализации генератора (по умолчанию 42)
        """
        np.random.seed(seed)
        logger.info(f"Установлен random seed: {seed}")
    
    @staticmethod
    def sine(vmin: float, vmax: float, samples: int, periods: float = 5, 
             duration: float = 1.0, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует синусоидальный сигнал.
        
        Args:
            vmin (float): Минимальное значение сигнала
            vmax (float): Максимальное значение сигнала
            samples (int): Количество отсчетов (игнорируется, если указан dt)
            periods (float): Количество периодов синусоиды (по умолчанию 5)
            duration (float): Длительность сигнала в секундах (по умолчанию 1.0)
            dt (float, optional): Шаг времени. Если указан, то игнорируется samples
            
        Returns:
            tuple: (t, v) - массивы времени и значений сигнала
        """
        if dt is not None:
            # Когда указан dt, генерируем временную ось с фиксированным шагом
            t = np.arange(0, duration, dt)
            samples = len(t)
        else:
            # Иначе генерируем равномерно распределенные точки
            t = np.linspace(0, duration, samples)
        
        # Расчет амплитуды и смещения
        A, D = (vmax-vmin)/2, (vmax+vmin)/2
        # Генерация синусоидального сигнала
        v = D + A*np.sin(2*np.pi*periods*t/duration)
        return t, v

    @staticmethod
    def step(vmin: float, vmax: float, samples: int, frac: float = 0.5, 
             duration: float = 1.0, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует ступенчатый сигнал.
        
        Args:
            vmin (float): Минимальное значение сигнала
            vmax (float): Максимальное значение сигнала
            samples (int): Количество отсчетов (игнорируется, если указан dt)
            frac (float): Доля времени (от 0 до 1), когда сигнал равен vmin (по умолчанию 0.5)
            duration (float): Длительность сигнала в секундах (по умолчанию 1.0)
            dt (float, optional): Шаг времени. Если указан, то игнорируется samples
            
        Returns:
            tuple: (t, v) - массивы времени и значений сигнала
        """
        if dt is not None:
            t = np.arange(0, duration, dt)
            samples = len(t)
        else:
            t = np.linspace(0, duration, samples)
        
        # Генерация ступенчатого сигнала
        v = np.where(t/duration < frac, vmin, vmax)
        return t, v

    @staticmethod
    def noise(vmin: float, vmax: float, samples: int, 
              duration: float = 1.0, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует случайный шум в заданном диапазоне.
        
        Args:
            vmin (float): Минимальное значение сигнала
            vmax (float): Максимальное значение сигнала
            samples (int): Количество отсчетов (игнорируется, если указан dt)
            duration (float): Длительность сигнала в секундах (по умолчанию 1.0)
            dt (float, optional): Шаг времени. Если указан, то игнорируется samples
            
        Returns:
            tuple: (t, v) - массивы времени и значений сигнала
        """
        if dt is not None:
            t = np.arange(0, duration, dt)
            samples = len(t)
        else:
            t = np.linspace(0, duration, samples)
        
        # Генерация случайного шума
        v = np.random.uniform(vmin, vmax, samples)
        return t, v

    @staticmethod
    def sweep(vmin: float, vmax: float, samples: int, 
              duration: float = 1.0, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует линейно возрастающий сигнал (развертку).
        
        Args:
            vmin (float): Начальное значение сигнала
            vmax (float): Конечное значение сигнала
            samples (int): Количество отсчетов (игнорируется, если указан dt)
            duration (float): Длительность сигнала в секундах (по умолчанию 1.0)
            dt (float, optional): Шаг времени. Если указан, то игнорируется samples
            
        Returns:
            tuple: (t, v) - массивы времени и значений сигнала
        """
        if dt is not None:
            t = np.arange(0, duration, dt)
            samples = len(t)
        else:
            t = np.linspace(0, duration, samples)
        
        # Генерация линейно возрастающего сигнала
        v = np.linspace(vmin, vmax, samples)
        return t, v

    @staticmethod
    def batch_variation(gen_func: Callable, base_params: Dict[str, float], 
                        variation: Dict[str, float], count: int, 
                        seed: Optional[int] = None) -> List[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """
        Возвращает список (params_i, t_i, v_i) с count экземплярами,
        params_i получаются из base_params с разбросом variation (fraction).
        
        Args:
            gen_func (callable): Функция генерации сигнала (один из методов класса)
            base_params (dict): Базовые параметры сигнала
            variation (dict): Диапазон вариации для каждого параметра ({param: frac, ...})
            count (int): Количество генерируемых экземпляров
            seed (int, optional): Seed для воспроизводимости (если None, не устанавливается)
            
        Returns:
            list: Список кортежей (params_i, t_i, v_i)
        """
        if seed is not None:
            SignalGenerator.set_random_seed(seed)
            
        data = []
        for i in range(count):
            # Генерация новых параметров с вариацией
            p = {
                k: v * (1 + np.random.uniform(-variation.get(k, 0), variation.get(k, 0)))
                for k, v in base_params.items()
            }
            
            # Гарантируем, что samples является целым числом
            if 'samples' in p:
                p['samples'] = int(p['samples'])
            
            # Вызов функции генерации с новыми параметрами
            t, v = gen_func(**p)
            data.append((p, t, v))
            
            logger.debug(f"Сгенерирована вариация {i+1}/{count} с параметрами: {p}")
            
        return data