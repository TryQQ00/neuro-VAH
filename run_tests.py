#!/usr/bin/env python3
"""
Простой скрипт для тестирования функциональности проекта.
Запускает базовые тесты на различных компонентах системы.
"""

import os
import sys
import unittest
import numpy as np
import torch
import logging
from matplotlib import pyplot as plt

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Импорт модулей проекта
try:
    from parser import VerilogAParser
    from generator import SignalGenerator
    from simulator import DeviceModel
    from spice_interface import SpiceSimulator
    from rnn_model import PhysicsInformedRNN, ModelTrainer
except ImportError as e:
    logger.error(f"Ошибка при импорте модулей: {e}")
    sys.exit(1)

class TestParser(unittest.TestCase):
    """
    Тесты для парсера Verilog-A файлов.
    """
    
    def setUp(self):
        self.parser = VerilogAParser()
        self.test_va_file = "examples/diode.va"
        
    def test_parser_initialization(self):
        """Проверка инициализации парсера"""
        self.assertIsNotNone(self.parser)
        
    def test_parse_diode_model(self):
        """Проверка парсинга модели диода"""
        if not os.path.exists(self.test_va_file):
            self.skipTest(f"Тестовый файл {self.test_va_file} не найден")
            
        result = self.parser.parse(self.test_va_file)
        self.assertIsNotNone(result)
        self.assertIn('module', result)
        self.assertEqual(result['module'], 'diode')
        self.assertIn('params', result)
        self.assertIn('Is', result['params'])
        self.assertIn('ports', result)
        self.assertIn('anode', result['ports'])
        self.assertIn('cathode', result['ports'])

class TestSignalGenerator(unittest.TestCase):
    """
    Тесты для генератора сигналов.
    """
    
    def test_sine_generation(self):
        """Проверка генерации синусоидального сигнала"""
        t, v = SignalGenerator.sine(vmin=-1, vmax=1, samples=100, periods=1)
        self.assertEqual(len(t), 100)
        self.assertEqual(len(v), 100)
        self.assertAlmostEqual(min(v), -1, delta=0.01)
        self.assertAlmostEqual(max(v), 1, delta=0.01)
        
    def test_step_generation(self):
        """Проверка генерации ступенчатого сигнала"""
        t, v = SignalGenerator.step(vmin=0, vmax=1, samples=100)
        self.assertEqual(len(t), 100)
        self.assertEqual(len(v), 100)
        self.assertEqual(v[0], 0)
        self.assertEqual(v[-1], 1)
        
    def test_sweep_generation(self):
        """Проверка генерации линейной развертки"""
        t, v = SignalGenerator.sweep(vmin=-2, vmax=2, samples=100)
        self.assertEqual(len(t), 100)
        self.assertEqual(len(v), 100)
        self.assertAlmostEqual(v[0], -2)
        self.assertAlmostEqual(v[-1], 2)
        
    def test_noise_generation(self):
        """Проверка генерации шумового сигнала"""
        t, v = SignalGenerator.noise(vmin=-1, vmax=1, samples=100)
        self.assertEqual(len(t), 100)
        self.assertEqual(len(v), 100)
        self.assertTrue(min(v) >= -1)
        self.assertTrue(max(v) <= 1)
        
class TestDeviceModel(unittest.TestCase):
    """
    Тесты для физических моделей устройств.
    """
    
    def test_diode_model(self):
        """Проверка модели диода"""
        params = {'Is': 1e-14, 'N': 1.0, 'Rs': 1.0}
        model = DeviceModel(params, 'Диод')
        
        # Генерация входного сигнала
        _, v = SignalGenerator.sweep(vmin=-0.1, vmax=0.7, samples=100)
        
        # Симуляция отклика
        i = model.simulate(v)
        
        self.assertEqual(len(i), 100)
        self.assertTrue(all(isinstance(x, float) for x in i))
        
        # Проверка монотонности ВАХ диода
        self.assertTrue(i[-1] > i[0])
        self.assertTrue(all(i[j] >= i[j-1] for j in range(1, len(i))))

class TestRNNModel(unittest.TestCase):
    """
    Тесты для RNN-модели.
    """
    
    def setUp(self):
        self.input_dim = 1
        self.hidden_dim = 32
        self.num_layers = 2
        self.dropout = 0.1
        self.model = PhysicsInformedRNN(self.hidden_dim, self.num_layers, self.dropout)
        
    def test_model_initialization(self):
        """Проверка инициализации модели"""
        self.assertIsNotNone(self.model)
        
    def test_forward_pass(self):
        """Проверка прямого прохода через модель"""
        batch_size = 2
        seq_len = 10
        
        # Создание тестовых входных данных
        x = torch.randn(batch_size, seq_len, self.input_dim)
        phys = torch.zeros(batch_size, seq_len, self.input_dim)
        
        # Прямой проход
        y = self.model(x, phys)
        
        # Проверка размерности выходных данных
        self.assertEqual(y.shape, (batch_size, seq_len, self.input_dim))

def run_tests():
    """
    Запуск всех тестов.
    """
    # Создание директории для примеров, если она не существует
    os.makedirs("examples", exist_ok=True)
    
    # Проверка наличия примеров Verilog-A файлов
    va_files = ["examples/diode.va", "examples/mosfet.va", "examples/memristor.va"]
    missing_files = [f for f in va_files if not os.path.exists(f)]
    
    if missing_files:
        logger.warning(f"Некоторые тестовые Verilog-A файлы отсутствуют: {missing_files}")
        logger.info("Эти файлы должны быть созданы для полного тестирования.")
    
    # Запуск тестов
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests() 