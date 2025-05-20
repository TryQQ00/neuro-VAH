import numpy as np
import logging
import traceback
from typing import Dict, Union, Tuple, Optional, Callable, List, Any

try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

logger = logging.getLogger(__name__)

class DeviceModel:
    """
    Физическая модель электронных компонентов с динамикой: диод, BJT, HEMT, мемристор.
    
    Поддерживает:
    - Симуляцию статической ВАХ
    - Эффекты самонагрева
    - Гистерезис
    - Различные методы численного интегрирования (Euler, RK2, RK4, адаптивный)
    - Векторизованные вычисления с ускорением Numba (при наличии)
    
    Attributes:
        p (dict): Параметры модели
        dev (str): Тип устройства ('Диод', 'BJT', 'HEMT', 'Memristor')
        T (float): Текущая температура устройства (K)
        H (float): Состояние гистерезиса (0-1)
        Rth (float): Тепловое сопротивление
        Cth (float): Тепловая ёмкость
        alphaT (float): Температурный коэффициент
        tauH (float): Постоянная времени гистерезиса
    """
    
    SUPPORTED_DEVICES = ['Диод']
    INTEGRATION_METHODS = ['euler', 'rk2', 'rk4', 'adaptive']
    
    def __init__(self, params: Dict[str, float], dev_type: str):
        """
        Инициализирует модель устройства.
        
        Args:
            params (dict): Словарь параметров компонента
            dev_type (str): Тип устройства ('Диод', 'BJT', 'HEMT', 'Memristor')
        
        Raises:
            ValueError: Если указан неподдерживаемый тип устройства
        """
        if dev_type not in self.SUPPORTED_DEVICES:
            raise ValueError(f"Неподдерживаемый тип устройства: {dev_type}")
            
        self.p = params.copy()  # Создаем копию параметров для безопасности
        self.dev = dev_type
        self.T = 300.0  # Начальная температура (K)
        self.H = 0.0    # Начальное состояние гистерезиса
        
        # Параметры тепловой модели
        self.Rth = params.get('Rth', 50.0)
        self.Cth = params.get('Cth', 1.0)
        self.alphaT = params.get('alphaT', 0.005)
        self.tauH = params.get('tauH', 0.1)
        
        # Для отладки
        logger.debug(f"Создана модель устройства {dev_type} с параметрами: {params}")
        
        # Создаем параметры в виде словаря, удобного для передачи в JIT-функции
        self._get_device_params()
        
    def _get_device_params(self) -> Dict[str, float]:
        """
        Извлекает актуальные параметры устройства в единый словарь для JIT-функций.
        
        Returns:
            Dict[str, float]: Словарь параметров для вычислений
        """
        # Базовые параметры
        device_params = {
            'device_type': self.SUPPORTED_DEVICES.index(self.dev),  # Преобразуем тип в число
            'T': self.T,
            'H': self.H,
            'Rth': self.Rth,
            'Cth': self.Cth,
            'alphaT': self.alphaT,
            'tauH': self.tauH,
            'beta_h': self.p.get('beta_h', 0.1)
        }
        
        # Параметры в зависимости от типа устройства
        if self.dev == 'Диод':
            device_params.update({
                'Is': self.p.get('Is', 1e-14),
                'N': self.p.get('N', 1.0),
                'Vt': 0.02585 * (self.T / 300)
            })
        elif self.dev == 'BJT':
            device_params.update({
                'IS': self.p.get('IS', 1e-15),  # Используем IS (большие буквы)
                'BF': self.p.get('BF', 100),    # Используем BF (большие буквы)
                'BR': self.p.get('BR', 1),      # Используем BR (большие буквы)
                'Vt': self.p.get('Vt', 0.02585 * (self.T / 300))
            })
        elif self.dev == 'HEMT':
            device_params.update({
                'VTH0': self.p.get('VTH0', 1.0),
                'Kp': self.p.get('Kp', 1.0)
            })
        elif self.dev == 'Memristor':
            device_params.update({
                'Ron': self.p.get('Ron', 100),
                'Roff': self.p.get('Roff', 16000),
                'a': self.p.get('a', 10),
                'v0': self.p.get('v0', 0.5)
            })
            
        return device_params
        
    def _static_current(self, v: float) -> float:
        """
        Вычисляет статический ток через устройство.
        
        Args:
            v (float): Напряжение
            
        Returns:
            float: Ток через устройство
        """
        if self.dev == 'Диод':
            Is = self.p.get('Is', 1e-14)
            n = self.p.get('N', 1.0)
            Vt = 0.02585 * (self.T / 300)
            return Is * (np.exp(v / (n * Vt)) - 1)
        return 0.0
        
    def _static_current_vector(self, v_arr: np.ndarray) -> np.ndarray:
        """
        Векторизованная версия _static_current для массивов напряжений.
        
        Args:
            v_arr (np.ndarray): Массив напряжений
            
        Returns:
            np.ndarray: Массив токов
        """
        # Получаем параметры устройства
        params = self._get_device_params()
        
        # Используем JIT-компиляцию, если доступна Numba и есть соответствующий метод
        if NUMBA_AVAILABLE and hasattr(DeviceModel, '_static_current_numba_vector'):
            try:
                # Пробуем использовать JIT-компилированную версию
                logger.debug("Пробуем использовать JIT-компилированную версию _static_current_numba_vector")
                return DeviceModel._static_current_numba_vector(v_arr, params)
            except Exception as e:
                # В случае ошибки логируем и используем обычную версию
                logger.warning(f"Ошибка в JIT-версии: {e}. Используем обычную реализацию.")
        
        # Используем обычную версию без JIT-компиляции
        logger.debug("Используем обычную векторизованную версию _static_current_vector")
        
        # Диодная модель
        if self.dev == 'Диод':
            Is = self.p.get('Is', 1e-14)
            n = self.p.get('N', 1.0)
            Vt = 0.02585 * (self.T / 300)
            return Is * (np.exp(v_arr / (n * Vt)) - 1)
            
        # Биполярный транзистор
        elif self.dev == 'BJT':
            Is = self.p.get('IS', 1e-15)
            beta = self.p.get('BF', 100)
            Vt = self.p.get('Vt', 0.02585 * (self.T / 300))
            ib = Is * (np.exp(v_arr / Vt) - 1)
            return beta * ib
            
        # HEMT
        elif self.dev == 'HEMT':
            Vth = self.p.get('VTH0', 1.0)
            Kp = self.p.get('Kp', 1.0)
            return np.where(v_arr <= Vth, 0.0, Kp * (v_arr - Vth) ** 2)
            
        # Мемристор
        elif self.dev == 'Memristor':
            Ron = self.p.get('Ron', 100)
            Roff = self.p.get('Roff', 16000)
            a = self.p.get('a', 10)
            v0 = self.p.get('v0', 0.5)
            G = Ron + (Roff - Ron) / (1 + np.exp(-a * (v_arr - v0)))
            return v_arr / G
            
        return np.zeros_like(v_arr)
    
    def _derive_state(self, state: np.ndarray, v: float, i: float, dt: float) -> np.ndarray:
        """
        Вычисляет производные состояния [dT/dt, dH/dt].
        
        Args:
            state (np.ndarray): Текущее состояние [T, H]
            v (float): Текущее напряжение
            i (float): Текущий ток
            dt (float): Шаг времени
            
        Returns:
            np.ndarray: Производные состояния [dT/dt, dH/dt]
        """
        T, H = state
        
        # Изменение температуры (самонагрев)
        P = v * i  # Мощность
        dT_dt = (P - (T - 300) / self.Rth) / self.Cth
        
        # Изменение гистерезиса
        Ith = self.p.get('Ith', 0)
        if abs(i) > Ith:
            dH_dt = (1.0 - H) / self.tauH  # Стремится к 1
        else:
            dH_dt = -H / self.tauH  # Стремится к 0
            
        return np.array([dT_dt, dH_dt])
    
    def _rk2_step(self, state: np.ndarray, v: float, dt: float) -> np.ndarray:
        """
        Выполняет шаг интегрирования методом Рунге-Кутты 2-го порядка.
        
        Args:
            state (np.ndarray): Текущее состояние [T, H]
            v (float): Текущее напряжение
            dt (float): Шаг времени
            
        Returns:
            np.ndarray: Новое состояние после интегрирования
        """
        T, H = state
        
        # Вычисление тока с текущими состояниями
        i0 = self._static_current(v)
        i0 *= 1 + self.alphaT * (T - 300)  # Учет температуры
        i0 *= (1 - self.p.get('beta_h', 0.1) * H)  # Учет гистерезиса
        
        # k1 - производные в начальной точке
        k1 = self._derive_state(state, v, i0, dt)
        
        # Промежуточное состояние в середине шага
        mid_state = state + 0.5 * dt * k1
        
        # Вычисление тока с промежуточными состояниями
        T_mid, H_mid = mid_state
        i_mid = self._static_current(v)
        i_mid *= 1 + self.alphaT * (T_mid - 300)
        i_mid *= (1 - self.p.get('beta_h', 0.1) * H_mid)
        
        # k2 - производные в середине шага
        k2 = self._derive_state(mid_state, v, i_mid, dt)
        
        # Новое состояние
        new_state = state + dt * k2
        
        return new_state
    
    def _rk4_step(self, state: np.ndarray, v: float, dt: float) -> np.ndarray:
        """
        Выполняет шаг интегрирования методом Рунге-Кутты 4-го порядка.
        
        Args:
            state (np.ndarray): Текущее состояние [T, H]
            v (float): Текущее напряжение
            dt (float): Шаг времени
            
        Returns:
            np.ndarray: Новое состояние после интегрирования
        """
        T, H = state
        
        # Функция для вычисления тока при данном состоянии
        def calc_current(s):
            T_s, H_s = s
            i_s = self._static_current(v)
            i_s *= 1 + self.alphaT * (T_s - 300)
            i_s *= (1 - self.p.get('beta_h', 0.1) * H_s)
            return i_s
        
        # k1
        i1 = calc_current(state)
        k1 = self._derive_state(state, v, i1, dt)
        
        # k2
        state2 = state + 0.5 * dt * k1
        i2 = calc_current(state2)
        k2 = self._derive_state(state2, v, i2, dt)
        
        # k3
        state3 = state + 0.5 * dt * k2
        i3 = calc_current(state3)
        k3 = self._derive_state(state3, v, i3, dt)
        
        # k4
        state4 = state + dt * k3
        i4 = calc_current(state4)
        k4 = self._derive_state(state4, v, i4, dt)
        
        # Итоговое состояние
        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return new_state
    
    def _adaptive_step(self, state: np.ndarray, v: float, dt: float, 
                      error_tol: float = 1e-4) -> Tuple[np.ndarray, float]:
        """
        Выполняет шаг с адаптивным размером.
        
        Args:
            state (np.ndarray): Текущее состояние [T, H]
            v (float): Текущее напряжение
            dt (float): Предлагаемый шаг времени
            error_tol (float): Допустимая относительная ошибка
            
        Returns:
            Tuple[np.ndarray, float]: Новое состояние и фактический использованный шаг
        """
        # Пробуем шаг h
        s1 = self._rk4_step(state, v, dt)
        
        # Делаем два полушага
        s_half = self._rk4_step(state, v, dt/2)
        s2 = self._rk4_step(s_half, v, dt/2)
        
        # Оцениваем ошибку (max по компонентам)
        err = np.max(np.abs((s2 - s1) / (np.abs(s1) + 1e-10)))
        
        if err < error_tol:
            # Ошибка меньше допустимой, возвращаем более точное s2
            new_dt = dt * min(2.0, max(0.5, 0.9 * (error_tol / err) ** 0.2))
            return s2, new_dt
        else:
            # Ошибка больше допустимой, уменьшаем шаг
            new_dt = dt * max(0.1, 0.9 * (error_tol / err) ** 0.25)
            # Ограничиваем глубину рекурсии для предотвращения stack overflow
            if dt < 1e-10:
                logger.warning(f"Достигнут минимальный шаг {dt}, ошибка всё ещё {err:.2e} > {error_tol:.2e}")
                return s2, dt  # Возвращаем результат несмотря на ошибку
            return self._adaptive_step(state, v, new_dt, error_tol)
    
    def simulate(self, v_arr: np.ndarray, dt: float = 1e-6) -> np.ndarray:
        """
        Производит симуляцию отклика устройства на заданное напряжение.
        
        Args:
            v_arr (np.ndarray): Массив напряжений
            dt (float): Шаг по времени (если None, используется 1мкс)
            
        Returns:
            np.ndarray: Массив токов
        """
        if not isinstance(v_arr, np.ndarray):
            v_arr = np.array(v_arr)
        if v_arr.ndim != 1 or v_arr.size == 0:
            raise ValueError("v_arr должен быть одномерным непустым массивом")
        i_arr = np.zeros_like(v_arr)
        state = np.array([self.T, 0.0])
        for idx, v in enumerate(v_arr):
            T, H = state
            i0 = self._static_current(v)
            i0 *= 1 + self.alphaT * (T - 300)
            i0 *= (1 - self.p.get('beta_h', 0.1) * H)
            i_arr[idx] = i0
            dT_dt = (v * i0 - (T - 300) / self.Rth) / self.Cth
            dH_dt = -H / self.tauH
            state = state + dt * np.array([dT_dt, dH_dt])
        return i_arr
    
    def simulate_static_iv(self, v_min: float, v_max: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует статическую ВАХ для устройства.
        
        Args:
            v_min (float): Минимальное напряжение развертки
            v_max (float): Максимальное напряжение развертки
            num_points (int): Количество точек в развертке
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Массивы напряжений и токов
        """
        # Генерируем линейную развертку напряжения
        v_arr = np.linspace(v_min, v_max, num_points)
        
        try:
            # Пробуем использовать векторизованную версию с JIT
            i_arr = self._static_current_vector(v_arr)
        except Exception as e:
            # В случае ошибки с JIT используем обычную векторизацию NumPy
            logger.warning(f"Ошибка при использовании JIT-векторизации: {e}. Используем обычный цикл.")
            i_arr = np.zeros_like(v_arr)
            for i, v in enumerate(v_arr):
                i_arr[i] = self._static_current(v)
        
        return v_arr, i_arr
    
    def reset_state(self):
        """Сбрасывает внутреннее состояние модели."""
        self.T = 300.0
        self.H = 0.0
        logger.debug("Состояние модели сброшено")

    def _simulate_euler(self, v_arr: np.ndarray, dt: float) -> np.ndarray:
        """
        Реализация метода Эйлера для симуляции отклика устройства.
        
        Args:
            v_arr (np.ndarray): Массив напряжений
            dt (float): Шаг времени
            
        Returns:
            np.ndarray: Массив токов
        """
        # Инициализируем выходной массив токов
        i_arr = np.zeros_like(v_arr)
        
        # Инициализируем состояние [T, H]
        state = np.array([self.T, 0.0])
        
        # Метод Эйлера
        for idx, v in enumerate(v_arr):
            # Вычисляем ток с текущими состояниями
            T, H = state
            i0 = self._static_current(v)
            i0 *= 1 + self.alphaT * (T - 300)
            i0 *= (1 - self.p.get('beta_h', 0.1) * H)
            i_arr[idx] = i0
            
            # Обновляем состояния (метод Эйлера)
            derivatives = self._derive_state(state, v, i0, dt)
            state = state + dt * derivatives
            
        return i_arr
        
    def _simulate_rk2(self, v_arr: np.ndarray, dt: float) -> np.ndarray:
        """
        Реализация метода Рунге-Кутты 2-го порядка для симуляции отклика устройства.
        
        Args:
            v_arr (np.ndarray): Массив напряжений
            dt (float): Шаг времени
            
        Returns:
            np.ndarray: Массив токов
        """
        # Инициализируем выходной массив токов
        i_arr = np.zeros_like(v_arr)
        
        # Инициализируем состояние [T, H]
        state = np.array([self.T, 0.0])
        
        # Пробуем использовать векторизованную JIT-версию
        if NUMBA_AVAILABLE and hasattr(DeviceModel, '_simulate_rk2_numba'):
            try:
                logger.debug("Пробуем использовать JIT-версию RK2")
                params = self._get_device_params()
                i_arr = DeviceModel._simulate_rk2_numba(v_arr, dt, params)
                logger.debug("Использована JIT-версия RK2")
                return i_arr
            except Exception as e:
                logger.warning(f"Ошибка в JIT-версии RK2: {e}. Используем обычную реализацию.")
        
        # Стандартная реализация RK2
        for idx, v in enumerate(v_arr):
            # Вычисляем ток с текущими состояниями
            T, H = state
            i0 = self._static_current(v)
            i0 *= 1 + self.alphaT * (T - 300)
            i0 *= (1 - self.p.get('beta_h', 0.1) * H)
            i_arr[idx] = i0
            
            # Обновляем состояния методом RK2
            state = self._rk2_step(state, v, dt)
            
        return i_arr
        
    def _simulate_rk4(self, v_arr: np.ndarray, dt: float) -> np.ndarray:
        """
        Реализация метода Рунге-Кутты 4-го порядка для симуляции отклика устройства.
        
        Args:
            v_arr (np.ndarray): Массив напряжений
            dt (float): Шаг времени
            
        Returns:
            np.ndarray: Массив токов
        """
        # Инициализируем выходной массив токов
        i_arr = np.zeros_like(v_arr)
        
        # Инициализируем состояние [T, H]
        state = np.array([self.T, 0.0])
        
        # Метод RK4
        for idx, v in enumerate(v_arr):
            # Вычисляем ток с текущими состояниями
            T, H = state
            i0 = self._static_current(v)
            i0 *= 1 + self.alphaT * (T - 300)
            i0 *= (1 - self.p.get('beta_h', 0.1) * H)
            i_arr[idx] = i0
            
            # Обновляем состояния методом RK4
            state = self._rk4_step(state, v, dt)
            
        return i_arr


# Если numba доступна, JIT-компилируем ключевые функции
if NUMBA_AVAILABLE:
    try:
        # Функция для вычисления статического тока (скалярная версия)
        @njit(cache=True)
        def _static_current_numba(v: float, params: Dict[str, float]) -> float:
            """JIT-компилированная версия _static_current"""
            device_type = int(params['device_type'])
            
            # Диодная модель
            if device_type == 0:  # 'Диод'
                Is = params.get('Is', 1e-14)
                n = params.get('N', 1.0)
                Vt = params.get('Vt', 0.02585)
                return Is * (np.exp(v / (n * Vt)) - 1)
                
            # Биполярный транзистор
            elif device_type == 1:  # 'BJT'
                Is = params.get('IS', 1e-15)
                beta = params.get('BF', 100)
                Vt = params.get('Vt', 0.02585)
                ib = Is * (np.exp(v / Vt) - 1)
                return beta * ib
                
            # HEMT
            elif device_type == 2:  # 'HEMT'
                Vth = params.get('VTH0', 1.0)
                Kp = params.get('Kp', 1.0)
                if v <= Vth:
                    return 0.0
                else:
                    return Kp * (v - Vth) ** 2
                
            # Мемристор
            elif device_type == 3:  # 'Memristor'
                Ron = params.get('Ron', 100)
                Roff = params.get('Roff', 16000)
                a = params.get('a', 10)
                v0 = params.get('v0', 0.5)
                G = Ron + (Roff - Ron) / (1 + np.exp(-a * (v - v0)))
                return v / G
                
            return 0.0
            
        # Векторизованная версия для массивов напряжений
        @njit(cache=True, parallel=True)
        def _static_current_numba_vector(v_arr: np.ndarray, params: Dict[str, float]) -> np.ndarray:
            """JIT-компилированная векторизованная версия _static_current"""
            result = np.zeros_like(v_arr)
            device_type = int(params['device_type'])
            
            # Оптимизированные версии для разных устройств
            if device_type == 0:  # 'Диод'
                Is = params.get('Is', 1e-14)
                n = params.get('N', 1.0)
                Vt = params.get('Vt', 0.02585)
                for i in prange(len(v_arr)):
                    result[i] = Is * (np.exp(v_arr[i] / (n * Vt)) - 1)
                    
            elif device_type == 1:  # 'BJT'
                Is = params.get('IS', 1e-15)
                beta = params.get('BF', 100)
                Vt = params.get('Vt', 0.02585)
                for i in prange(len(v_arr)):
                    ib = Is * (np.exp(v_arr[i] / Vt) - 1)
                    result[i] = beta * ib
                    
            elif device_type == 2:  # 'HEMT'
                Vth = params.get('VTH0', 1.0)
                Kp = params.get('Kp', 1.0)
                for i in prange(len(v_arr)):
                    if v_arr[i] <= Vth:
                        result[i] = 0.0
                    else:
                        result[i] = Kp * (v_arr[i] - Vth) ** 2
                        
            elif device_type == 3:  # 'Memristor'
                Ron = params.get('Ron', 100)
                Roff = params.get('Roff', 16000)
                a = params.get('a', 10)
                v0 = params.get('v0', 0.5)
                for i in prange(len(v_arr)):
                    G = Ron + (Roff - Ron) / (1 + np.exp(-a * (v_arr[i] - v0)))
                    result[i] = v_arr[i] / G
                    
            return result
            
        # Полная JIT-компилированная версия симуляции с RK2
        @njit(cache=True)
        def _simulate_rk2_numba(v_arr: np.ndarray, dt: float, params: Dict[str, float]) -> np.ndarray:
            """JIT-компилированная версия simulate с методом RK2"""
            # Инициализация
            i_arr = np.zeros_like(v_arr)
            state = np.array([300.0, 0.0])  # [T, H]
            
            # Извлекаем параметры
            T_ambient = 300.0
            alphaT = params.get('alphaT', 0.005)
            Rth = params.get('Rth', 50.0)
            Cth = params.get('Cth', 1.0)
            beta_h = params.get('beta_h', 0.1)
            tauH = params.get('tauH', 0.1)
            Ith = params.get('Ith', 0)
            
            # Основной цикл интегрирования
            for idx in range(len(v_arr)):
                v = v_arr[idx]
                T, H = state
                
                # Вычисляем ток с текущими состояниями
                i0 = _static_current_numba(v, params)
                i0 *= 1 + alphaT * (T - T_ambient)
                i0 *= (1 - beta_h * H)
                i_arr[idx] = i0
                
                # RK2 шаг
                # k1 - производные в начальной точке
                P = v * i0  # Мощность
                dT_dt = (P - (T - T_ambient) / Rth) / Cth
                if abs(i0) > Ith:
                    dH_dt = (1.0 - H) / tauH
                else:
                    dH_dt = -H / tauH
                k1 = np.array([dT_dt, dH_dt])
                
                # Промежуточное состояние
                mid_state = state + 0.5 * dt * k1
                T_mid, H_mid = mid_state
                
                # Вычисляем ток для промежуточного состояния
                i_mid = _static_current_numba(v, params)
                i_mid *= 1 + alphaT * (T_mid - T_ambient)
                i_mid *= (1 - beta_h * H_mid)
                
                # k2 - производные в середине шага
                P_mid = v * i_mid
                dT_dt_mid = (P_mid - (T_mid - T_ambient) / Rth) / Cth
                if abs(i_mid) > Ith:
                    dH_dt_mid = (1.0 - H_mid) / tauH
                else:
                    dH_dt_mid = -H_mid / tauH
                k2 = np.array([dT_dt_mid, dH_dt_mid])
                
                # Обновляем состояние
                state = state + dt * k2
                
            return i_arr
            
        # Регистрируем JIT-функции как методы класса
        DeviceModel._static_current_numba = staticmethod(_static_current_numba)
        DeviceModel._static_current_numba_vector = staticmethod(_static_current_numba_vector)
        DeviceModel._simulate_rk2_numba = staticmethod(_simulate_rk2_numba)
        
        logger.info("Numba JIT-компиляция успешно применена к методам DeviceModel")
        
    except Exception as e:
        logger.warning(f"Не удалось применить Numba JIT-компиляцию: {e}")
        logger.debug(traceback.format_exc())

# Генерация сигналов
class SignalGenerator:
    @staticmethod
    def sweep(vmin: float, vmax: float, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if samples < 2 or vmin >= vmax:
            raise ValueError("Некорректные параметры sweep")
        t = np.linspace(0, 1, samples)
        v = np.linspace(vmin, vmax, samples)
        return t, v
    @staticmethod
    def sine(vmin: float, vmax: float, samples: int, periods: float = 1) -> Tuple[np.ndarray, np.ndarray]:
        if samples < 2 or vmin >= vmax or periods <= 0:
            raise ValueError("Некорректные параметры sine")
        t = np.linspace(0, 1, samples)
        A, D = (vmax-vmin)/2, (vmax+vmin)/2
        v = D + A * np.sin(2 * np.pi * periods * t)
        return t, v