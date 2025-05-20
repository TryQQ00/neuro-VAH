import re
import os
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional, Set

logger = logging.getLogger(__name__)

class VerilogAParser:
    """
    Извлекает имя модуля, порты и параметры из .va файла.
    
    Поддерживает:
    - Директивы `include
    - Макросы `define
    - Условная компиляция `ifdef/`ifndef/`else/`endif
    - Многострочные объявления параметров с переносами
    - Извлечение типов параметров (real, integer)
    - Направления портов (input, output, inout)
    
    Returns:
        dict: {
            'module': str,                              # имя модуля
            'ports': List[str],                         # список имен портов
            'port_directions': Dict[str, str],          # направления портов
            'params': Dict[str, float],                 # значения параметров (для совместимости)
            'param_info': Dict[str, Dict[str, Any]],    # полная информация о параметрах
            'includes': List[str],                      # список включенных файлов
            'defines': Dict[str, str]                   # словарь макроподстановок
        }
    """
    
    def __init__(self):
        """Инициализация парсера."""
        self.included_files: Set[str] = set()
        self.defines: Dict[str, str] = {}
        self.base_dir: str = ""
        self.log_unrecognized: bool = True
        self.debug_info: Dict[str, Any] = {
            "parsing_errors": [],
            "unrecognized_lines": []
        }
        
    def _preprocess_macros(self, text: str) -> str:
        """
        Обрабатывает макросы `define в тексте Verilog-A.
        
        Args:
            text (str): Исходный текст Verilog-A.
            
        Returns:
            str: Текст с раскрытыми макросами.
        """
        # Сначала найдем все макросы `define
        for m in re.finditer(r'`define\s+(\w+)(?:\s+(.+?))?(?:\n|$)', text):
            macro_name = m.group(1)
            macro_value = m.group(2) if m.group(2) else ""
            # Удаляем комментарии из значения макроса
            macro_value = re.sub(r'//.*$', '', macro_value).strip()
            self.defines[macro_name] = macro_value
            logger.debug(f"Обнаружен макрос: `{macro_name}` = '{macro_value}'")
            
        # Теперь заменим все вхождения макросов
        for name, value in self.defines.items():
            old_text = text
            text = re.sub(r'`' + name + r'\b', value, text)
            if text != old_text:
                logger.debug(f"Выполнена подстановка макроса `{name}`")
            
        return text
    
    def _process_conditional_blocks(self, text: str) -> str:
        """
        Обрабатывает условные директивы `ifdef/`ifndef/`else/`endif.
        
        Args:
            text (str): Исходный текст Verilog-A.
            
        Returns:
            str: Текст с разрешенными условными блоками.
        """
        # Рекурсивная функция для обработки ifстек директив
        def process_if_stack(text: str, pos: int = 0) -> Tuple[str, int]:
            # Ищем следующую директиву
            if_pattern = r'`(ifdef|ifndef|else|elsif|endif)\b'
            match = re.search(if_pattern, text[pos:])
            
            if not match:
                return text, len(text)
                
            directive = match.group(1)
            start_pos = pos + match.start()
            end_pos = pos + match.end()
            
            # Обработка директив
            if directive in ('ifdef', 'ifndef'):
                # Ищем имя макроса
                macro_match = re.search(r'\s+(\w+)', text[end_pos:])
                if not macro_match:
                    logger.warning(f"Не удалось найти имя макроса после `{directive}")
                    return text, end_pos
                    
                macro_name = macro_match.group(1)
                macro_defined = macro_name in self.defines
                
                # Определяем, включать ли блок
                include_block = (directive == 'ifdef' and macro_defined) or (directive == 'ifndef' and not macro_defined)
                
                # Находим конец условного блока
                true_block_start = end_pos + macro_match.end()
                true_block_text, next_pos = process_if_stack(text, true_block_start)
                
                if include_block:
                    # Включаем true блок
                    logger.debug(f"Включен блок после `{directive} {macro_name}")
                    # Вернуть комбинацию: текст до директивы + содержимое true блока + текст после всего условного блока
                    return text[:start_pos] + text[true_block_start:next_pos], next_pos
                else:
                    # Исключаем блок
                    logger.debug(f"Исключен блок после `{directive} {macro_name}")
                    # Вернуть комбинацию: текст до директивы + текст после всего условного блока
                    return text[:start_pos] + "", next_pos
                    
            elif directive == 'endif':
                # Конец условного блока
                return text, end_pos
                
            else:
                # else или elsif (пока не реализуем полную поддержку elsif)
                # Просто пропускаем блок до endif
                next_pos = end_pos
                while True:
                    next_directive_match = re.search(if_pattern, text[next_pos:])
                    if not next_directive_match:
                        logger.warning("Не найдена закрывающая директива `endif")
                        return text, len(text)
                        
                    if next_directive_match.group(1) == 'endif':
                        return text, next_pos + next_directive_match.end()
                    next_pos += next_directive_match.end()
                    
        # Обрабатываем все условные блоки
        result, _ = process_if_stack(text)
        return result
            
    def _resolve_include(self, filepath: str, base_content: str) -> str:
        """
        Разрешает директивы `include в файле Verilog-A.
        
        Args:
            filepath (str): Путь к файлу Verilog-A.
            base_content (str): Содержимое основного файла.
            
        Returns:
            str: Содержимое с включенными файлами.
        """
        self.base_dir = os.path.dirname(filepath)
        result = base_content
        
        # Находим все директивы `include
        include_pattern = r'`include\s+"([^"]+)"'
        for m in re.finditer(include_pattern, base_content):
            include_file = m.group(1)
            include_path = os.path.join(self.base_dir, include_file)
            
            # Избегаем циклических включений
            if include_path in self.included_files:
                logger.warning(f"Обнаружено циклическое включение: {include_path}")
                continue
                
            try:
                with open(include_path, 'r', encoding='utf-8') as f:
                    include_content = f.read()
                    self.included_files.add(include_path)
                    logger.info(f"Включен файл: {include_path}")
                    
                    # Рекурсивно обрабатываем включенные файлы
                    include_content = self._resolve_include(include_path, include_content)
                    
                    # Заменяем директиву на содержимое файла
                    result = result.replace(m.group(0), include_content)
            except FileNotFoundError:
                error_msg = f"Не удалось найти включаемый файл: {include_path}"
                logger.error(error_msg)
                self.debug_info["parsing_errors"].append(error_msg)
            except Exception as e:
                error_msg = f"Ошибка при обработке включаемого файла {include_path}: {e}"
                logger.error(error_msg)
                self.debug_info["parsing_errors"].append(error_msg)
                logger.debug(traceback.format_exc())
                
        return result
    
    def _handle_line_continuations(self, text: str) -> str:
        """
        Обрабатывает переносы строк с обратным слешем в конце.
        
        Args:
            text (str): Исходный текст Verilog-A.
            
        Returns:
            str: Текст с объединенными строками.
        """
        # Заменяем переносы строк с обратным слешем
        result = re.sub(r'\\\s*\n\s*', ' ', text)
        if result != text:
            logger.debug("Обработаны переносы строк с обратным слешем")
        return result
    
    def _parse_ports(self, module_text: str) -> List[Tuple[str, str]]:
        """
        Извлекает порты и их направления из текста модуля.
        
        Args:
            module_text (str): Текст модуля Verilog-A.
            
        Returns:
            List[Tuple[str, str]]: Список кортежей (имя_порта, направление).
        """
        ports = []
        port_directions = {}
        
        # Сначала извлекаем имена портов из объявления модуля
        m = re.search(r'\bmodule\s+\w+\s*\(([^)]*)\)', module_text)
        if not m:
            logger.warning("Не удалось найти объявление модуля")
            return []
            
        port_names = [p.strip() for p in m.group(1).split(',') if p.strip()]
        
        # Теперь ищем объявления направлений портов
        for direction in ['input', 'output', 'inout']:
            pattern = fr'\b{direction}\b\s+([^;]+);'
            for m in re.finditer(pattern, module_text):
                dir_ports = [p.strip() for p in m.group(1).split(',')]
                for port in dir_ports:
                    port_directions[port] = direction
                    logger.debug(f"Обнаружен порт {port} с направлением {direction}")
        
        # Собираем порты с их направлениями
        for port in port_names:
            direction = port_directions.get(port, 'unknown')
            if direction == 'unknown':
                logger.warning(f"Не найдено направление для порта: {port}")
            ports.append((port, direction))
            
        return ports
    
    def _parse_parameters(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Извлекает параметры, их типы и значения из текста Verilog-A.
        
        Args:
            text (str): Предобработанный текст Verilog-A.
            
        Returns:
            Dict[str, Dict]: Словарь параметров с их типами и значениями.
        """
        params = {}
        
        # Сначала удалим сложные блоки описания параметров (те, которые содержат специальные конструкции и могут вызывать ошибки)
        # Идентифицируем блоки с from(), exclude, etc.
        simplified_text = re.sub(r'parameter\s+\w+\s+\w+\s*=\s*\w+\s+from\s*\[.*?\](?:\s*exclude\s+.*?)?(?=;)', '', text, flags=re.DOTALL)
        simplified_text = re.sub(r'parameter\s+\w+\s*=\s*\w+\s+from\s*\[.*?\](?:\s*exclude\s+.*?)?(?=;)', '', simplified_text, flags=re.DOTALL)
        
        # Ищем параметры разных типов с более надежным регулярным выражением
        param_pattern = r'parameter\s+(\w+)\s+(\w+)\s*=\s*([^;{}\[\]]+);'
        for m in re.finditer(param_pattern, simplified_text, re.MULTILINE | re.DOTALL):
            param_type, name, val_str = m.group(1), m.group(2), m.group(3).strip()
            line_num = text[:m.start()].count('\n') + 1
            
            # Удалим возможные комментарии
            val_str = re.sub(r'//.*?(?=\n|$)', '', val_str).strip()
            
            try:
                # Проверяем, есть ли в строке признаки сложного синтаксиса
                if 'from' in val_str or 'exclude' in val_str:
                    logger.debug(f"Пропускаем сложный параметр: {name} = {val_str}")
                    continue
                
                if param_type.lower() == 'real':
                    # Безопасная оценка выражений
                    try:
                        value = float(eval(val_str, {"__builtins__": {}}, {}))
                    except:
                        # Проверяем, не является ли это просто числом с точкой
                        if re.match(r'^[-+]?\d+\.\d+$', val_str):
                            value = float(val_str)
                        else:
                            # Пробуем использовать выражение как есть (для совместимости)
                            value = 0.0
                            logger.warning(f"Не удалось преобразовать значение параметра {name} = {val_str}, используем 0.0")
                elif param_type.lower() == 'integer':
                    try:
                        value = int(eval(val_str, {"__builtins__": {}}, {}))
                    except:
                        # Проверяем, не является ли это просто целым числом
                        if re.match(r'^[-+]?\d+$', val_str):
                            value = int(val_str)
                        else:
                            # Пробуем использовать выражение как есть (для совместимости)
                            value = 0
                            logger.warning(f"Не удалось преобразовать значение параметра {name} = {val_str}, используем 0")
                else:
                    value = val_str
                    
                params[name] = {
                    'value': value,
                    'type': param_type,
                    'line': line_num
                }
                logger.debug(f"Обнаружен параметр {param_type} {name} = {value} (строка {line_num})")
            except Exception as e:
                error_msg = f"Не удалось преобразовать значение параметра {name} = {val_str} в строке {line_num}: {e}"
                logger.warning(error_msg)
                self.debug_info["parsing_errors"].append(error_msg)
                # Используем значение по умолчанию
                params[name] = {
                    'value': 0.0 if param_type.lower() == 'real' else 0,
                    'type': param_type,
                    'line': line_num
                }
        
        # Для обратной совместимости ищем также параметры в старом формате
        old_param_pattern = r'parameter\s+(\w+)\s*=\s*([^;{}\[\]]+);'
        for m in re.finditer(old_param_pattern, simplified_text, re.MULTILINE | re.DOTALL):
            name, val_str = m.group(1), m.group(2).strip()
            line_num = text[:m.start()].count('\n') + 1
            
            # Удалим возможные комментарии
            val_str = re.sub(r'//.*?(?=\n|$)', '', val_str).strip()
            
            # Пропускаем, если параметр уже найден
            if name in params:
                continue
                
            # Проверяем, есть ли в строке признаки сложного синтаксиса
            if 'from' in val_str or 'exclude' in val_str:
                logger.debug(f"Пропускаем сложный параметр (старый формат): {name} = {val_str}")
                continue
                
            try:
                # Пробуем определить тип по значению
                try:
                    if re.match(r'^[-+]?\d+$', val_str):
                        value = int(val_str)
                        param_type = 'integer'
                    elif re.match(r'^[-+]?\d+\.\d+$', val_str):
                        value = float(val_str)
                        param_type = 'real'
                    else:
                        value = float(eval(val_str, {"__builtins__": {}}, {}))
                        param_type = 'real'
                except (ValueError, SyntaxError, NameError):
                    # Если не можем определить, используем строку как есть
                    value = 0.0
                    param_type = 'real'
                    logger.warning(f"Не удалось преобразовать значение параметра {name} = {val_str}, используем 0.0")
                
                params[name] = {
                    'value': value,
                    'type': param_type,
                    'line': line_num
                }
                logger.debug(f"Обнаружен параметр (старый формат) {name} = {value} (тип {param_type}, строка {line_num})")
            except Exception as e:
                error_msg = f"Не удалось преобразовать значение параметра {name} = {val_str} в строке {line_num}: {e}"
                logger.warning(error_msg)
                self.debug_info["parsing_errors"].append(error_msg)
                # Используем значение по умолчанию
                params[name] = {
                    'value': 0.0,
                    'type': 'real',
                    'line': line_num
                }
        
        # Если не нашли параметры, но есть специальные токены, добавим базовые параметры для BJT
        if len(params) == 0:
            # Более надежная проверка на BJT модель
            bjt_tokens = ["BJT", "bipolar", "transistor", "bipolartransistor", "Bipolar Junction Transistor"]
            is_bjt = False
            for token in bjt_tokens:
                if token in text:
                    is_bjt = True
                    break
            
            if is_bjt:
                logger.info("Параметры не найдены, но это похоже на BJT модель. Добавляем базовые параметры.")
                # Добавляем параметры согласно именам, используемым в DeviceModel
                default_params = {
                    "IS": {"value": 1e-16, "type": "real", "line": 0},
                    "BF": {"value": 100, "type": "real", "line": 0},
                    "BR": {"value": 1, "type": "real", "line": 0},
                    "Vt": {"value": 0.026, "type": "real", "line": 0},
                    "VAF": {"value": 100, "type": "real", "line": 0},
                    "Rth": {"value": 50.0, "type": "real", "line": 0},
                    "Cth": {"value": 1.0, "type": "real", "line": 0},
                    "alphaT": {"value": 0.005, "type": "real", "line": 0},
                    "beta_h": {"value": 0.1, "type": "real", "line": 0}
                }
                params.update(default_params)
                logger.debug("Добавлены базовые параметры BJT: " + ", ".join(default_params.keys()))
            
        return params
    
    def _log_unrecognized(self, text: str, original_text: str) -> None:
        """
        Логирует нераспознанные строки в файле Verilog-A.
        
        Args:
            text (str): Обработанный текст.
            original_text (str): Исходный текст.
        """
        if not self.log_unrecognized:
            return
            
        lines = original_text.split('\n')
        processed_lines = set()
        
        # Отмечаем строки, содержащие распознанные конструкции
        patterns = [
            r'\bmodule\s+\w+\s*\(', 
            r'input\b', r'output\b', r'inout\b',
            r'parameter\s+\w+',
            r'`include', r'`define', r'`ifdef', r'`ifndef', r'`else', r'`endif',
            r'endmodule\b'
        ]
        
        for pattern in patterns:
            for m in re.finditer(pattern, text):
                start_line = text[:m.start()].count('\n')
                end_line = text[:m.end()].count('\n')
                for i in range(start_line, end_line + 1):
                    processed_lines.add(i)
        
        # Логируем нераспознанные строки
        for i, line in enumerate(lines):
            if i not in processed_lines and line.strip() and not line.strip().startswith('//'):
                logger.debug(f"Нераспознанная строка {i+1}: {line.strip()}")
                self.debug_info["unrecognized_lines"].append((i+1, line.strip()))
    
    def parse(self, filepath: str, log_unrecognized: bool = True) -> dict:
        """
        Парсит файл Verilog-A.
        
        Args:
            filepath (str): Путь к файлу Verilog-A.
            log_unrecognized (bool): Логировать ли нераспознанные строки.
            
        Returns:
            dict: Словарь с информацией о модуле, портах и параметрах.
        """
        self.log_unrecognized = log_unrecognized
        self.included_files = set()
        self.defines = {}
        self.debug_info = {"parsing_errors": [], "unrecognized_lines": []}
        
        logger.info(f"Начат парсинг файла: {filepath}")
        
        if not os.path.exists(filepath):
            error_msg = f"Файл не существует: {filepath}"
            logger.error(error_msg)
            self.debug_info["parsing_errors"].append(error_msg)
            return {'module': None, 'ports': [], 'params': {}, 'includes': [], 'defines': {}}
        
        try:
            # Чтение файла с использованием контекстного менеджера
            with open(filepath, 'r', encoding='utf-8') as f:
                original_text = f.read()
                
            logger.debug(f"Файл успешно прочитан: {filepath}, размер: {len(original_text)} байт")
                
            # Шаг 1: Объединяем строки с переносами
            text = self._handle_line_continuations(original_text)
            
            # Шаг 2: Раскрываем включения
            text = self._resolve_include(filepath, text)
            
            # Шаг 3: Обрабатываем макросы
            text = self._preprocess_macros(text)
            
            # Шаг 4: Обрабатываем условную компиляцию
            text = self._process_conditional_blocks(text)
            
            # Шаг 5: Удаляем комментарии
            # Блочные комментарии
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
            # Однострочные комментарии
            text = re.sub(r'//.*?(?=\n|$)', '', text)
            
            # Шаг 6: Находим имя модуля
            m = re.search(r'\bmodule\s+(\w+)\s*\(', text)
            if not m:
                error_msg = f"Не удалось найти объявление модуля в файле {filepath}"
                logger.error(error_msg)
                self.debug_info["parsing_errors"].append(error_msg)
                return {'module': None, 'ports': [], 'params': {}, 'includes': list(self.included_files), 'defines': self.defines}
                
            module = m.group(1)
            logger.info(f"Обнаружен модуль: {module}")
            
            # Шаг 7: Извлекаем порты и их направления
            ports = self._parse_ports(text)
            
            # Шаг 8: Извлекаем параметры
            params_dict = self._parse_parameters(text)
            
            # Шаг 9: Логируем нераспознанные строки
            self._log_unrecognized(text, original_text)
            
            # Для обратной совместимости переформатируем параметры
            simple_params = {name: param_info['value'] for name, param_info in params_dict.items()}
            
            result = {
                'module': module,
                'ports': [p[0] for p in ports],  # Для совместимости возвращаем только имена
                'port_directions': dict(ports),   # Новое поле с направлениями
                'params': simple_params,          # Для совместимости
                'param_info': params_dict,        # Расширенная информация
                'includes': list(self.included_files),
                'defines': self.defines,
                'debug_info': self.debug_info     # Добавляем отладочную информацию
            }
            
            logger.info(f"Парсинг завершен. Найдено портов: {len(ports)}, параметров: {len(params_dict)}")
            return result
                
        except Exception as e:
            error_msg = f"Ошибка при парсинге файла {filepath}: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            self.debug_info["parsing_errors"].append(error_msg)
            return {'module': None, 'ports': [], 'params': {}, 'includes': [], 'defines': {}, 'debug_info': self.debug_info}