import re
import os
from typing import Dict, List, Any, Union

class VerilogAParser:
    """
    Парсер Verilog-A: имя модуля, порты, параметры (даже если справа макрос или выражение).
    Игнорирует define/include/комментарии. Если параметр не вычисляется — сохраняет как строку.
    """
    def parse(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Удаляем define/include/комментарии
        clean_lines = []
        for line in lines:
            s = line.strip()
            if not s or s.startswith('//') or s.startswith('`define') or s.startswith('`include'):
                continue
            # Удаляем однострочные комментарии
            s = re.sub(r'//.*', '', s)
            clean_lines.append(s)
        text = '\n'.join(clean_lines)
        # Имя модуля
        m = re.search(r'\bmodule\s+(\w+)\s*\(([^)]*)\)', text)
        if not m:
            raise ValueError("Не найдено объявление модуля")
        module = m.group(1)
        port_str = m.group(2)
        ports = [p.strip() for p in port_str.split(',') if p.strip()]
        # Параметры (любые parameter ... = ...;)
        params = {}
        for pm in re.finditer(r'parameter\s+(real|integer)?\s*(\w+)\s*=\s*([^;]+);', text):
            _typ, name, val = pm.groups()
            val = val.strip()
            # Пробуем вычислить значение, иначе оставляем строкой
            try:
                if _typ == 'real':
                    value = float(eval(val, {"__builtins__": {}}, {}))
                elif _typ == 'integer':
                    value = int(eval(val, {"__builtins__": {}}, {}))
                else:
                    # Без типа — пробуем float, потом int
                    try:
                        value = float(eval(val, {"__builtins__": {}}, {}))
                    except Exception:
                        value = int(eval(val, {"__builtins__": {}}, {}))
            except Exception:
                value = val  # Сохраняем как строку (например, макрос)
            params[name] = value
        if not params:
            raise ValueError("Не найдено ни одного параметра")
        return {'module': module, 'ports': ports, 'params': params}

# Юнит-тест (создаётся файл tests/test_parser.py)