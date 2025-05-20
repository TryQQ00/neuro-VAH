import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import random
from typing import Callable, Dict, Tuple, List, Optional, Union
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class PhysicsInformedDataset(Dataset):
    """
    Ленивый датасет для расчета физических токов на лету.
    
    Attributes:
        V (torch.Tensor): Тензор напряжений [batch, seq_len, 1]
        I (torch.Tensor): Тензор измеренных токов [batch, seq_len, 1]
        phys_func (Callable): Функция для расчета физических токов
    """
    def __init__(self, V: torch.Tensor, I: torch.Tensor, phys_func: Callable):
        """
        Инициализирует датасет.
        
        Args:
            V (torch.Tensor): Тензор напряжений [batch, seq_len, 1]
            I (torch.Tensor): Тензор измеренных токов [batch, seq_len, 1]
            phys_func (Callable): Функция для расчета физических токов
        """
        self.V = V
        self.I = I
        self.phys_func = phys_func
        self.phys_cache = {}  # Кэш для уже вычисленных значений
        
    def __len__(self) -> int:
        return len(self.V)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает элемент датасета с расчетом физического тока на лету.
        
        Args:
            idx (int): Индекс элемента
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (V, I_phys, I)
        """
        v = self.V[idx]
        i = self.I[idx]
        
        # Вычисляем физический ток только если его еще нет в кэше
        if idx not in self.phys_cache:
            v_np = v.squeeze(-1).cpu().numpy()
            i_phys_np = self.phys_func(v_np)
            
            # Убедимся, что i_phys_np имеет правильную форму перед созданием тензора
            if isinstance(i_phys_np, np.ndarray):
                # Проверка размерности массива
                if i_phys_np.ndim == 1:  # [seq_len]
                    # Проверяем, совпадает ли длина массива с ожидаемой
                    if len(i_phys_np) == v.size(0):
                        i_phys = torch.tensor(i_phys_np, dtype=torch.float32).unsqueeze(-1)
                    else:
                        # Если длины не совпадают, заполняем нулями
                        logger.warning(f"Длина выхода phys_func ({len(i_phys_np)}) не совпадает с длиной входа ({v.size(0)})")
                        i_phys = torch.zeros_like(v)
                elif i_phys_np.ndim == 2:  # [batch или другое, seq_len]
                    # Если первая размерность равна 1, это может быть батч
                    if i_phys_np.shape[0] == 1:
                        i_phys = torch.tensor(i_phys_np[0], dtype=torch.float32).unsqueeze(-1)
                    else:
                        # Иначе берем последний элемент как seq_len
                        i_phys = torch.tensor(i_phys_np, dtype=torch.float32)
                        if i_phys.size(0) != v.size(0):
                            logger.warning(f"Размерность выхода phys_func ({i_phys_np.shape}) не соответствует входу ({v.shape})")
                            i_phys = torch.zeros_like(v)
                        i_phys = i_phys.unsqueeze(-1)
                elif i_phys_np.ndim >= 3:  # [batch, seq_len, 1 или более]
                    # Берем первый элемент батча и первый канал
                    if i_phys_np.shape[0] == 1:
                        i_phys = torch.tensor(i_phys_np[0, :, 0], dtype=torch.float32).unsqueeze(-1)
                    else:
                        logger.warning(f"Неожиданная размерность выхода phys_func: {i_phys_np.shape}")
                        i_phys = torch.zeros_like(v)
                else:
                    # Это может быть скаляр в numpy (0-мерный массив)
                    i_phys_value = i_phys_np.item() if i_phys_np.size == 1 else 0
                    i_phys = torch.full_like(v, i_phys_value)
            else:
                # Если это скаляр, заполняем им весь тензор
                i_phys_value = float(i_phys_np) if np.isscalar(i_phys_np) else 0
                i_phys = torch.full_like(v, i_phys_value)
            
            self.phys_cache[idx] = i_phys
        else:
            i_phys = self.phys_cache[idx]
            
        return v, i_phys, i


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

def train_model(model, device, X, Y, epochs=50, lr=1e-3, batch_size=32, save_plot_path=None):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if save_plot_path:
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(save_plot_path)
    return model, losses

class EarlyStopping:
    """
    Механизм ранней остановки обучения при отсутствии улучшений.
    
    Attributes:
        patience (int): Количество эпох без улучшений до остановки
        delta (float): Минимальное изменение для признания улучшения
        best_score (float): Лучшее значение метрики
        counter (int): Счетчик эпох без улучшений
        early_stop (bool): Флаг остановки обучения
    """
    def __init__(self, patience: int = 5, delta: float = 0.0):
        """
        Инициализирует механизм ранней остановки.
        
        Args:
            patience (int): Количество эпох без улучшений до остановки
            delta (float): Минимальное изменение для признания улучшения
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module, path: str):
        """
        Обрабатывает текущее значение функции потерь.
        
        Args:
            val_loss (float): Текущее значение функции потерь на валидации
            model (nn.Module): Текущая модель
            path (str): Путь для сохранения лучшей модели
            
        Returns:
            bool: True, если обучение следует остановить, иначе False
        """
        score = -val_loss  # Больше score -> лучше модель
        
        if self.best_score is None:
            # Первый запуск
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # Нет улучшения
            self.counter += 1
            logger.info(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Есть улучшение
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
        return self.early_stop
    
    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        """
        Сохраняет текущую лучшую модель.
        
        Args:
            val_loss (float): Текущее значение функции потерь
            model (nn.Module): Текущая модель
            path (str): Путь для сохранения модели
        """
        logger.info(f'Улучшение валидационной потери: {val_loss:.6f}. Сохраняем модель...')
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)


class ModelTrainer:
    """
    Обучение и оценка модели с поддержкой физической функции.
    
    Attributes:
        model (PhysicsInformedRNN): Модель для обучения
        device (torch.device): Устройство для вычислений (GPU/CPU)
        phys_func (Callable): Функция для расчета физических токов
        criterion (nn.Module): Функция потерь
        optimizer (optim.Optimizer): Оптимизатор
        scheduler (optim.lr_scheduler._LRScheduler): Планировщик скорости обучения
        checkpoint_dir (str): Директория для сохранения чекпоинтов
    """
    def __init__(
        self, 
        model: LSTMModel, 
        device: torch.device, 
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Инициализирует тренера модели.
        
        Args:
            model (PhysicsInformedRNN): Модель для обучения
            device (torch.device): Устройство для вычислений (GPU/CPU)
            learning_rate (float): Скорость обучения
            weight_decay (float): Коэффициент L2-регуляризации
            checkpoint_dir (str): Директория для сохранения чекпоинтов
        """
        self.model = model.to(device)
        self.dev = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        self.checkpoint_dir = checkpoint_dir
        
        # Создаем директорию для чекпоинтов
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Для логирования
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def set_seed(self, seed: int = 42):
        """
        Устанавливает seed для воспроизводимости результатов.
        
        Args:
            seed (int): Значение seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Установлен seed: {seed}")
        
    def train(
        self, 
        Vseq: torch.Tensor, 
        Iseq: torch.Tensor, 
        epochs: int = 50, 
        batch_size: int = 32,
        val_split: float = 0.2,
        patience: int = 5,
        seed: Optional[int] = 42
    ) -> Dict[str, List[float]]:
        """
        Обучает модель с разделением на train/validation и ранней остановкой.
        
        Args:
            Vseq (torch.Tensor): Тензор напряжений [batch, seq_len, 1]
            Iseq (torch.Tensor): Тензор измеренных токов [batch, seq_len, 1]
            epochs (int): Максимальное количество эпох
            batch_size (int): Размер мини-батча
            val_split (float): Доля данных для валидации (0-1)
            patience (int): Количество эпох без улучшений до остановки
            seed (int): Значение seed для воспроизводимости
            
        Returns:
            Dict[str, List[float]]: Словарь с историей потерь train/val
        """
        if seed is not None:
            self.set_seed(seed)
            
        # Создаем датасет
        dataset = PhysicsInformedDataset(Vseq, Iseq, self.phys_func)
        
        # Разделяем на train/validation
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed if seed is not None else 42)
        )
        
        # Создаем даталоадеры
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Инициализируем раннюю остановку
        early_stopping = EarlyStopping(patience=patience)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        
        # Очищаем историю обучения
        self.train_losses = []
        self.val_losses = []
        
        # Обучение
        logger.info(f"Начинаем обучение на {train_size} примерах, валидация на {val_size} примерах")
        logger.info(f"Устройство: {self.dev}, Эпохи: {epochs}, Batch Size: {batch_size}")
        
        for epoch in range(epochs):
            # Режим обучения
            self.model.train()
            train_loss = 0.0
            
            for Vb, Ip, Ib in train_loader:
                # Перемещаем данные на устройство
                Vb, Ip, Ib = Vb.to(self.dev), Ip.to(self.dev), Ib.to(self.dev)
                
                # Обнуляем градиенты
                self.optimizer.zero_grad()
                
                # Прямой проход
                pred = self.model(Vb)
                
                # Вычисляем потери
                loss = self.criterion(pred, Ib)
                
                # Обратный проход и оптимизация
                loss.backward()
                self.optimizer.step()
                
                # Учитываем потери
                train_loss += loss.item()
            
            # Средние потери за эпоху
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Режим оценки
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for Vb, Ip, Ib in val_loader:
                    # Перемещаем данные на устройство
                    Vb, Ip, Ib = Vb.to(self.dev), Ip.to(self.dev), Ib.to(self.dev)
                    
                    # Прямой проход
                    pred = self.model(Vb)
                    
                    # Вычисляем потери
                    loss = self.criterion(pred, Ib)
                    
                    # Учитываем потери
                    val_loss += loss.item()
            
            # Средние потери за эпоху
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            # Обновляем планировщик
            self.scheduler.step(val_loss)
            
            # Выводим прогресс
            logger.info(f"Эпоха {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Проверяем раннюю остановку
            if early_stopping(val_loss, self.model, checkpoint_path):
                logger.info(f"Ранняя остановка на эпохе {epoch+1}")
                break
        
        # Загружаем лучшую модель
        if os.path.exists(checkpoint_path):
            logger.info(f"Загружаем лучшую модель из {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def evaluate(self, Vseq: torch.Tensor, Iseq: torch.Tensor) -> Dict[str, float]:
        """
        Оценивает модель на тестовых данных.
        
        Args:
            Vseq (torch.Tensor): Тензор напряжений [batch, seq_len, 1]
            Iseq (torch.Tensor): Тензор измеренных токов [batch, seq_len, 1]
            
        Returns:
            Dict[str, float]: Словарь с метриками оценки
        """
        self.model.eval()
        
        with torch.no_grad():
            # Вычисляем физический ток
            v_np = Vseq.squeeze(-1).cpu().numpy()
            i_phys_np = self.phys_func(v_np)
            
            # Проверяем и корректируем форму выходных данных
            if isinstance(i_phys_np, np.ndarray):
                # Анализируем размерность результата
                if i_phys_np.ndim == 3:  # [batch, seq_len, 1]
                    # Правильная форма, просто преобразуем в тензор
                    phys = torch.tensor(i_phys_np, dtype=torch.float32).to(self.dev)
                elif i_phys_np.ndim == 2:  # [batch, seq_len] или [seq_len, something]
                    # Определяем, что это за размерности
                    if i_phys_np.shape[0] == Vseq.size(0):  # [batch, seq_len]
                        phys = torch.tensor(i_phys_np, dtype=torch.float32).unsqueeze(-1).to(self.dev)
                    else:  # [seq_len, something]
                        # Предполагаем, что первая размерность - sequence_length
                        logger.warning(f"Неожиданная форма выхода phys_func: {i_phys_np.shape}, ожидалось примерно {Vseq.shape}")
                        # Преобразуем с повторением для каждого элемента в batch
                        try:
                            if i_phys_np.shape[0] == Vseq.size(1):  # [seq_len, ?]
                                # Повторяем для каждого batch
                                phys = torch.tensor(i_phys_np, dtype=torch.float32)
                                # Убираем лишние размерности, если они есть
                                if phys.ndim > 2:
                                    phys = phys[:, 0]
                                phys = phys.unsqueeze(0).repeat(Vseq.size(0), 1).unsqueeze(-1).to(self.dev)
                            else:
                                # Форма несовместима, создаем нулевой тензор
                                phys = torch.zeros_like(Vseq).to(self.dev)
                        except Exception as e:
                            logger.error(f"Ошибка при обработке выхода phys_func: {e}")
                            phys = torch.zeros_like(Vseq).to(self.dev)
                elif i_phys_np.ndim == 1:  # [seq_len]
                    # Проверяем, совпадает ли длина с ожидаемой
                    if len(i_phys_np) == Vseq.size(1):
                        # Повторяем для каждого batch
                        phys = torch.tensor(i_phys_np, dtype=torch.float32)
                        phys = phys.unsqueeze(0).repeat(Vseq.size(0), 1).unsqueeze(-1).to(self.dev)
                    else:
                        logger.warning(f"Длина выхода phys_func ({len(i_phys_np)}) не соответствует ожидаемой ({Vseq.size(1)})")
                        phys = torch.zeros_like(Vseq).to(self.dev)
                else:
                    # Скаляр или другая форма, заполняем всё одним значением
                    i_phys_value = i_phys_np.item() if i_phys_np.size == 1 else 0
                    phys = torch.full_like(Vseq, i_phys_value).to(self.dev)
            else:
                # Скалярный результат
                try:
                    i_phys_value = float(i_phys_np)
                    phys = torch.full_like(Vseq, i_phys_value).to(self.dev)
                except:
                    logger.error(f"Невозможно преобразовать выход phys_func к числу: {i_phys_np}")
                    phys = torch.zeros_like(Vseq).to(self.dev)
            
            # Перемещаем данные на устройство
            Vseq, Iseq = Vseq.to(self.dev), Iseq.to(self.dev)
            
            # Прямой проход
            pred = self.model(Vseq)
            
            # Вычисляем метрики
            mse = self.criterion(pred, Iseq).item()
            mae = nn.L1Loss()(pred, Iseq).item()
            
            # Коэффициент детерминации (R²)
            var = torch.var(Iseq)
            r2 = 1 - torch.sum((pred - Iseq) ** 2) / (len(Iseq.flatten()) * var)
            
            # Возвращаем метрики
            return {
                'MSE': mse,
                'MAE': mae,
                'R2': r2.item()
            }
    
    def save_onnx(self, path: str, seq_len: int = 100, dynamic_axes: bool = True):
        """
        Экспортирует модель в формат ONNX.
        
        Args:
            path (str): Путь для сохранения модели
            seq_len (int): Длина последовательности для dummy-входов
            dynamic_axes (bool): Использовать ли динамические оси
        """
        # Переводим модель в режим оценки
        self.model.eval()
        
        # Создаем dummy-входы
        Vdummy = torch.randn(1, seq_len, 1).to(self.dev)
        Idummy = torch.randn(1, seq_len, 1).to(self.dev)
        
        # Определяем динамические оси
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'V': {0: 'batch_size', 1: 'sequence_length'},
                'Iphys': {0: 'batch_size', 1: 'sequence_length'},
                'Ipred': {0: 'batch_size', 1: 'sequence_length'}
            }
        
        # Экспортируем модель
        try:
            torch.onnx.export(
                self.model,
                Vdummy,
                path,
                input_names=['V'],
                output_names=['Ipred'],
                dynamic_axes=dynamic_axes_dict,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                verbose=False
            )
            logger.info(f"Модель успешно экспортирована в ONNX: {path}")
            
            # Проверяем экспортированную модель
            try:
                import onnx
                model = onnx.load(path)
                onnx.checker.check_model(model)
                logger.info("ONNX-модель проверена успешно")
            except ImportError:
                logger.warning("Библиотека onnx не установлена. Пропускаем проверку модели.")
            except Exception as e:
                logger.error(f"Ошибка при проверке ONNX-модели: {e}")
                
        except Exception as e:
            logger.error(f"Ошибка при экспорте в ONNX: {e}")
            raise