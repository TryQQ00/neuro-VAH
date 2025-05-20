import re, subprocess, tempfile, os
import logging
import traceback
import numpy as np
import io
import shutil
import signal
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator, TextIO

logger = logging.getLogger(__name__)

try:
    import PySpice
    from PySpice.Spice.Parser import SpiceParser
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Unit import *
    PYSPICE_AVAILABLE = True
    logger.info("PySpice найден и будет использован для анализа результатов")
except ImportError:
    PYSPICE_AVAILABLE = False
    logger.warning("PySpice не найден. Будет использован встроенный парсер.")

# Таймаут для ngspice в секундах
NGSPICE_TIMEOUT = 30

@contextmanager
def timeout_manager(seconds: int) -> Iterator[None]:
    """
    Контекстный менеджер для ограничения времени выполнения операции.
    
    Args:
        seconds (int): Таймаут в секундах
        
    Raises:
        TimeoutError: Если операция превысила таймаут
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Операция превысила таймаут {seconds} секунд")
        
    # Установка обработчика сигнала
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        # Устанавливаем таймер
        signal.alarm(seconds)
        yield
    finally:
        # Сбрасываем таймер
        signal.alarm(0)
        # Восстанавливаем оригинальный обработчик
        signal.signal(signal.SIGALRM, original_handler)

class SpiceSimulator:
    """
    Интерфейс для запуска ngspice и анализа результатов.
    
    Поддерживает:
    - Запуск SPICE-симуляций через ngspice
    - Переходные анализы (tran)
    - Парсинг результатов из ASCII .raw файлов
    - Многопортовые схемы
    - Безопасную работу с временными файлами
    - Ограничение времени выполнения симуляции
    """
    
    @staticmethod
    def run(va_fp: str, module: str, ports: List[str], params: Dict[str, float], v_arr: np.ndarray,
            dt: float, cleanup: bool = True, 
            timeout: int = NGSPICE_TIMEOUT) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполняет SPICE-симуляцию и возвращает результаты.
        
        Args:
            va_fp (str): Путь к Verilog-A файлу
            module (str): Имя модуля
            ports (List[str]): Список портов
            params (Dict[str, float]): Параметры устройства
            v_arr (np.ndarray): Массив входных напряжений
            dt (float): Шаг времени. Должен быть явно указан
            cleanup (bool): Удалять ли временные файлы после выполнения
            timeout (int): Таймаут выполнения ngspice в секундах
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Массивы времени и тока
            
        Raises:
            subprocess.CalledProcessError: Если ngspice завершается с ошибкой
            TimeoutError: Если ngspice выполняется дольше timeout
            ValueError: Если не удается распарсить данные или результаты некорректны
            FileNotFoundError: Если файл Verilog-A или ngspice не найдены
        """
        # Проверяем, что dt явно указан
        if dt is None:
            dt = 1e-6  # Шаг времени по умолчанию (1 мкс)
            logger.warning(f"dt не указан, используется значение по умолчанию: {dt} сек")
        
        # Проверяем установку ngspice
        if not SpiceSimulator.check_ngspice_installed():
            raise FileNotFoundError("ngspice не найден. Убедитесь, что он установлен и доступен в PATH.")
        
        # Проверяем существование файла Verilog-A
        if not os.path.exists(va_fp):
            raise FileNotFoundError(f"Файл Verilog-A не найден: {va_fp}")
        
        # Абсолютный путь к файлу (для корректной работы с временными директориями)
        va_fp_abs = os.path.abspath(va_fp)
        
        # Создаем временную директорию внутри контекстного менеджера
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Формируем пути к временным файлам
                cir_path = os.path.join(temp_dir, "temp_circuit.cir")
                raw_path = os.path.join(temp_dir, "temp_circuit.raw")
                err_path = os.path.join(temp_dir, "ngspice_error.log")
                
                # Создаем netlist
                netlist = SpiceSimulator._create_netlist(
                    va_fp_abs, module, ports, params, v_arr, dt
                )
                
                # Записываем netlist во временный файл
                with open(cir_path, 'w') as f:
                    f.write(netlist)
                    
                logger.debug(f"Создан временный netlist: {cir_path}")
                
                # Запускаем ngspice с перехватом ошибок и ограничением времени
                try:
                    with timeout_manager(timeout):
                        result = subprocess.run(
                            ['ngspice', '-b', cir_path], 
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        logger.debug(f"ngspice выполнен успешно")
                        
                        # Сохраняем stdout и stderr для отладки
                        if result.stdout.strip():
                            logger.debug(f"ngspice stdout: {result.stdout[:500]}...")
                        if result.stderr.strip():
                            # Записываем полный stderr в файл для отладки
                            with open(err_path, 'w') as f:
                                f.write(result.stderr)
                            logger.warning(f"ngspice stderr (сохранен в {err_path}): {result.stderr[:200]}...")
                            
                except TimeoutError as e:
                    logger.error(f"Превышено время выполнения ngspice ({timeout} сек)")
                    # Попытка завершить процесс (если он еще работает)
                    try:
                        subprocess.run(['pkill', '-f', f'ngspice.*{cir_path}'], check=False)
                    except Exception:
                        pass
                    raise
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"Ошибка при выполнении ngspice (код {e.returncode})")
                    
                    # Записываем вывод и ошибки в файлы для отладки
                    with open(err_path, 'w') as f:
                        f.write(f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
                        
                    # Формируем информативное сообщение об ошибке
                    error_msg = f"ngspice завершился с ошибкой (код {e.returncode})\n"
                    if e.stderr:
                        error_lines = e.stderr.splitlines()
                        error_msg += "Ошибки:\n" + "\n".join(error_lines[:10])
                        if len(error_lines) > 10:
                            error_msg += f"\n... и еще {len(error_lines)-10} строк (см. {err_path})"
                    
                    # Если файл с результатами не создан, копируем .cir для отладки
                    if not os.path.exists(raw_path) and not cleanup:
                        backup_cir = os.path.join(os.path.dirname(va_fp_abs), "failed_circuit.cir")
                        try:
                            shutil.copy(cir_path, backup_cir)
                            error_msg += f"\nNetlist сохранен в {backup_cir}"
                        except Exception:
                            pass
                            
                    raise subprocess.CalledProcessError(
                        e.returncode, e.cmd, e.output, error_msg
                    ) from e
                    
                # Парсим результаты
                if PYSPICE_AVAILABLE:
                    # Используем PySpice для парсинга
                    try:
                        time_array, current_array = SpiceSimulator._parse_with_pyspice(raw_path, ports)
                        return time_array, current_array
                    except Exception as e:
                        logger.warning(f"Не удалось использовать PySpice: {e}. Переключаемся на ручной парсинг.")
                        logger.debug(traceback.format_exc())
                        
                # Если PySpice не доступен или не сработал, используем ручной парсинг
                if os.path.exists(raw_path):
                    return SpiceSimulator._parse_raw(raw_path)
                else:
                    error_msg = f"Файл результатов не найден: {raw_path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                    
            except Exception as e:
                logger.error(f"Ошибка при выполнении SPICE-симуляции: {e}")
                # Если необходимо, сохраняем временные файлы для отладки
                if not cleanup:
                    backup_dir = os.path.join(os.path.dirname(va_fp_abs), "spice_debug")
                    os.makedirs(backup_dir, exist_ok=True)
                    try:
                        for f in os.listdir(temp_dir):
                            src = os.path.join(temp_dir, f)
                            dst = os.path.join(backup_dir, f)
                            if os.path.isfile(src):
                                shutil.copy2(src, dst)
                        logger.info(f"Временные файлы сохранены в {backup_dir} для отладки")
                    except Exception as backup_err:
                        logger.warning(f"Не удалось сохранить временные файлы: {backup_err}")
                
                # Перебрасываем исключение дальше
                raise
    
    @staticmethod
    def _create_netlist(va_fp: str, module: str, ports: List[str], 
                       params: Dict[str, float], v_arr: np.ndarray, 
                       dt: float) -> str:
        """
        Создает SPICE netlist для симуляции.
        
        Args:
            va_fp (str): Путь к Verilog-A файлу
            module (str): Имя модуля
            ports (List[str]): Список портов
            params (Dict[str, float]): Параметры устройства
            v_arr (np.ndarray): Массив входных напряжений
            dt (float): Шаг времени
            
        Returns:
            str: SPICE netlist
        """
        lines = []
        lines.append('* Автоматически сгенерированный SPICE netlist')
        lines.append('.option rawfmt=ascii')
        lines.append('.option reltol=1e-4 abstol=1e-9')  # Улучшенные настройки погрешности
        
        # Добавляем параметры с округлением для избежания экспоненциальной записи для малых чисел
        for k, v in params.items():
            # Определяем формат числа в зависимости от его порядка
            if abs(v) < 1e-6 or abs(v) > 1e6:
                val_str = f"{v:.10e}"
            else:
                val_str = f"{v:.10f}".rstrip('0').rstrip('.')
            lines.append(f'.param {k}={val_str}')
        
        # Создаем PWL (кусочно-линейный) источник напряжения с оптимизацией для больших массивов
        if len(v_arr) > 1000:
            # Для больших массивов сохраняем PWL в отдельный файл
            pwl_file = 'pwl_data.txt'
            pwl_lines = []
            for i, val in enumerate(v_arr):
                pwl_lines.append(f"{i*dt:.12e} {val:.12e}")
            
            pwl_content = '\n'.join(pwl_lines)
            
            # Убедимся, что файл будет в той же директории, где будет выполняться ngspice
            lines.append(f'* PWL данные находятся в файле {pwl_file}')
            
            # Этот файл будет создан позже вместе с .cir файлом
            pwl = f'PWL(FILE="{pwl_file}")'
        else:
            # Для небольших массивов включаем PWL напрямую
            pwl_points = []
            for i, val in enumerate(v_arr):
                pwl_points.append(f"{i*dt:.12e} {val:.12e}")
            pwl = 'PWL(' + ' '.join(pwl_points) + ')'
        
        # Подключаем источники в зависимости от количества портов
        if len(ports) == 2:
            # Двухпортовое устройство (диод, мемристор)
            lines.append(f'V1 {ports[0]} {ports[1]} {pwl}')
        elif len(ports) == 3:
            # Трехпортовое устройство (транзистор)
            lines.append(f'Vgs {ports[1]} {ports[2]} {pwl}')  # Затвор
            lines.append(f'Vds {ports[0]} {ports[2]} {pwl}')  # Сток
        else:
            # Многопортовое устройство
            logger.warning(f"Нестандартное устройство с {len(ports)} портами")
            for i in range(len(ports) - 1):
                lines.append(f'V{i} {ports[i]} {ports[-1]} {pwl}')
        
        # Включаем .va файл и добавляем компонент
        lines.append(f'.include "{va_fp}"')
        node_list = ' '.join(ports)
        lines.append(f"XU1 {node_list} {module}")
        
        # Настраиваем анализ
        t_end = len(v_arr) * dt
        lines.append(f'.tran {dt} {t_end}')
        
        # Определяем, какие токи мерять
        if len(ports) == 2:
            lines.append('.print tran I(V1)')
        elif len(ports) == 3:
            lines.append('.print tran I(Vgs) I(Vds)')
        else:
            output_list = ' '.join(f'I(V{i})' for i in range(len(ports) - 1))
            lines.append(f'.print tran {output_list}')
            
        lines.append('.end')
        
        # Соединяем все строки в netlist
        netlist = '\n'.join(lines)
        
        return netlist
    
    @staticmethod
    def _parse_with_pyspice(raw_file: str, ports: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Парсит результаты симуляции с помощью PySpice.
        
        Args:
            raw_file (str): Путь к .raw файлу
            ports (List[str]): Список портов для определения имени переменной тока
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Массивы времени и тока
        """
        raw_data = PySpice.Spice.RawFile.RawFile(raw_file)
        
        # Получаем временную ось
        time_array = raw_data.variables['time'].data
        
        # Определяем переменную тока в зависимости от количества портов
        if len(ports) == 2:
            current_var = 'i(v1)'
        elif len(ports) == 3:
            current_var = 'i(vds)'  # Ток стока для транзистора
        else:
            current_var = f'i(v0)'  # Первый ток для многопортового устройства
        
        # Получаем массив токов, приводим имя к нижнему регистру для совместимости
        current_var = current_var.lower()
        
        # Ищем переменную в списке доступных переменных
        available_vars = [v.lower() for v in raw_data.variables.keys()]
        if current_var not in available_vars:
            logger.warning(f"Переменная {current_var} не найдена. Доступные переменные: {available_vars}")
            # Попробуем найти любую переменную тока
            current_vars = [v for v in available_vars if v.startswith('i(')]
            if current_vars:
                current_var = current_vars[0]
                logger.info(f"Используем доступную переменную тока: {current_var}")
            else:
                raise ValueError(f"Не найдено ни одной переменной тока в результатах симуляции")
        
        current_array = raw_data.variables[current_var].data
        
        return time_array, current_array
    
    @staticmethod
    def _parse_raw(raw_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Парсит ASCII .raw файл вручную.
        
        Args:
            raw_file (str): Путь к .raw файлу
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Массивы времени и тока
        """
        try:
            # Объявляем массивы данных
            time_values = []
            current_values = []
            
            # Чтение файла построчно, что более эффективно для больших файлов
            with open(raw_file, 'r') as f:
                # Сначала ищем начало блока данных
                in_data_section = False
                for line in f:
                    if in_data_section:
                        # Парсим строку с данными
                        parts = line.split()
                        if len(parts) >= 3:  # Индекс, время, значение(я)
                            try:
                                time = float(parts[1])
                                current = float(parts[2])
                                time_values.append(time)
                                current_values.append(current)
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Пропущена строка из-за ошибки: {line.strip()} ({e})")
                    elif line.strip() == 'Values:':
                        # Нашли начало секции данных
                        in_data_section = True
            
            # Проверяем, что у нас есть данные
            if not time_values:
                # Если данных нет, попробуем альтернативный метод на основе регулярных выражений
                with open(raw_file, 'r') as f:
                    content = f.read()
                
                # Находим секцию с данными
                data_section = re.search(r'Values:\s*\n(.+?)(?:\n\n|\Z)', content, re.DOTALL)
                if not data_section:
                    raise ValueError("Секция с данными не найдена в .raw файле")
                    
                # Извлекаем только строки с данными
                data_lines = data_section.group(1).strip().split('\n')
                
                # Обрабатываем каждую строку
                for line in data_lines:
                    parts = line.split()
                    if len(parts) >= 3:  # Индекс, время, значение(я)
                        try:
                            time = float(parts[1])
                            current = float(parts[2])
                            time_values.append(time)
                            current_values.append(current)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Пропущена строка из-за ошибки: {line} ({e})")
                
            # Проверяем, что у нас есть данные
            if not time_values:
                raise ValueError("Не удалось извлечь данные из .raw файла")
                
            # Конвертируем в numpy массивы
            time_array = np.array(time_values)
            current_array = np.array(current_values)
            
            return time_array, current_array
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге .raw файла {raw_file}: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    @staticmethod
    def check_ngspice_installed() -> bool:
        """
        Проверяет, установлен ли ngspice.
        
        Returns:
            bool: True, если ngspice найден, иначе False
        """
        try:
            result = subprocess.run(['ngspice', '--version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True,
                                   check=False,
                                   timeout=5)  # Ограничиваем время выполнения
                                   
            if result.returncode == 0:
                version_text = result.stdout.strip()
                logger.info(f"Обнаружен ngspice: {version_text}")
                return True
            else:
                logger.error(f"ngspice вернул ненулевой код: {result.returncode}")
                if result.stderr:
                    logger.error(f"Ошибка: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Таймаут при проверке ngspice")
            return False
        except FileNotFoundError:
            logger.error("ngspice не найден. Убедитесь, что он установлен и доступен в PATH.")
            return False
        except Exception as e:
            logger.error(f"Ошибка при проверке ngspice: {e}")
            logger.debug(traceback.format_exc())
            return False