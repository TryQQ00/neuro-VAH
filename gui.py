import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
from parser import VerilogAParser
from simulator import DeviceModel
from generator import SignalGenerator
from rnn_model import LSTMModel, train_model, ModelTrainer
import torch
import os
from loss import iv_loss

def draw_plot(x, y, filename='plot.png', xlabel='X', ylabel='Y', title=''):
    plt.figure(figsize=(7,3))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def draw_iv_plot(v, i_phys, i_pred, filename='iv_plot.png'):
    plt.figure(figsize=(7,3))
    plt.plot(v, i_phys, label='Reference')
    plt.plot(v, i_pred, '--', label='Predicted')
    plt.xlabel('V')
    plt.ylabel('I')
    plt.legend()
    plt.title('I-V Curve')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main_gui():
    sg.theme('LightBlue3')
    layout = [
        [sg.Text('Verilog-A файл:'), sg.Input(key='-VAFILE-', size=(60,1)), sg.FileBrowse(file_types=(('VA','*.va'),))],
        [sg.Text('Тип устройства:'), sg.Combo(['Диод', 'MOSFET', 'BJT'], default_value='Диод', key='-DEVTYPE-', readonly=True)],
        [sg.Text('Vmin:'), sg.Input('0.0', key='-VMIN-', size=(8,1)),
         sg.Text('Vmax:'), sg.Input('1.0', key='-VMAX-', size=(8,1)),
         sg.Text('Samples:'), sg.Input('100', key='-SAMPLES-', size=(8,1)),
         sg.Text('Epochs:'), sg.Input('50', key='-EPOCHS-', size=(8,1))],
        [sg.Text('Vgs (MOSFET):'), sg.Input('2.0', key='-VGS-', size=(8,1)),  # Новое поле
         sg.Text('Vbe (BJT):'), sg.Input('0.7', key='-VBE-', size=(8,1))],    # Новое поле
        [sg.Button('Запустить обучение', key='-RUN-')],
        [sg.Text('Loss (MSE):')],
        [sg.Image(key='-LOSS-')],
        [sg.Text('I-V Curve:')],
        [sg.Image(key='-IV-')],
        [sg.Text('', key='-ERROR-', size=(100,2), text_color='red')]
    ]
    window = sg.Window("neuro-VAH", layout, finalize=True)
    window.set_min_size((800, 600))
    window.TKroot.minsize(900, 600)

    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, None):
            break
        if event == '-RUN-':
            try:
                va_file = values['-VAFILE-']
                dev_type = values['-DEVTYPE-']  # Выбор типа устройства вручную
                vmin = float(values['-VMIN-'])
                vmax = float(values['-VMAX-'])
                samples = int(values['-SAMPLES-'])
                epochs = int(values['-EPOCHS-'])
                vgs = float(values['-VGS-'])  # Новое поле
                vbe = float(values['-VBE-'])  # Новое поле
                if samples < 2:
                    raise ValueError('Samples должно быть не менее 2')
                parser = VerilogAParser()
                info = parser.parse(va_file)
                params = info['params']
                # Для MOSFET и BJT добавляем управляющее напряжение в параметры
                if dev_type == 'MOSFET':
                    params['Vgs'] = vgs
                if dev_type == 'BJT':
                    params['Vbe'] = vbe
                t, v = SignalGenerator.sweep(vmin, vmax, samples)
                device = DeviceModel(params, dev_type)
                i_phys = device.simulate(v)
                # Проверка на NaN/inf
                if not np.all(np.isfinite(i_phys)):
                    raise ValueError('В симуляции физической модели получены некорректные значения (NaN/inf). Проверьте параметры и диапазон напряжений.')

                # --- Нормализация данных ---
                v_min, v_max = v.min(), v.max()
                i_min, i_max = i_phys.min(), i_phys.max()
                v_norm = (v - v_min) / (v_max - v_min + 1e-12)
                i_norm = (i_phys - i_min) / (i_max - i_min + 1e-12)
                X = torch.tensor(v_norm[None, :, None], dtype=torch.float32)
                Y = torch.tensor(i_norm[None, :, None], dtype=torch.float32)

                # --- Обучение с ModelTrainer ---
                model = LSTMModel(hidden_size=128, num_layers=3, dropout=0.3)
                device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                def phys_func(v_np):
                    v_np = np.asarray(v_np)
                    if v_np.ndim == 1:
                        return (device.simulate(v_np) - i_min) / (i_max - i_min + 1e-12)
                    elif v_np.ndim == 2:
                        return np.stack([(device.simulate(row) - i_min) / (i_max - i_min + 1e-12) for row in v_np])
                    else:
                        raise ValueError("phys_func: v_np должен быть 1D или 2D массивом")
                trainer = ModelTrainer(model, device_torch, learning_rate=5e-4, checkpoint_dir='results')
                trainer.phys_func = phys_func

                val_split = 0.0  # Для одиночного батча всегда отключаем валидацию

                history = trainer.train(X, Y, epochs=epochs, batch_size=32, val_split=val_split, patience=20, seed=42, phys_reg_alpha=0.1)
                losses = history['train_losses']

                # --- Генерация данных из .va (пример) ---
                # Здесь используйте реальную физическую модель или ngspice, а не np.sin(V_seq)
                # Например:
                # from simulator import DeviceModel
                # device = DeviceModel(params, 'Диод')
                # I_seq = device.simulate(V_seq)

                # График loss (EMA сглаживание)
                try:
                    def ema(arr, alpha=0.1):
                        out = []
                        for i, v in enumerate(arr):
                            if i == 0:
                                out.append(v)
                            else:
                                out.append(alpha * v + (1 - alpha) * out[-1])
                        return out
                    smoothed_losses = ema(losses, alpha=0.1)
                    loss_png = 'loss_plot.png'
                    draw_plot(np.arange(len(smoothed_losses)), smoothed_losses, filename=loss_png, xlabel='Epoch', ylabel='Loss', title='Training Loss (Smoothed)')
                    window['-LOSS-'].update(filename=loss_png)
                except Exception as e:
                    window['-ERROR-'].update(f'Ошибка построения графика Loss: {e}')

                # Сравнение ВАХ
                try:
                    model.eval()
                    with torch.no_grad():
                        y_pred_norm = model(X.to(device_torch)).cpu().numpy().reshape(-1)
                    y_pred = y_pred_norm * (i_max - i_min) + i_min
                    if not np.all(np.isfinite(y_pred)):
                        raise ValueError('В предсказании нейросети получены некорректные значения (NaN/inf).')
                    iv_png = 'iv_plot.png'
                    draw_iv_plot(v, i_phys, y_pred, filename=iv_png)
                    window['-IV-'].update(filename=iv_png)
                except Exception as e:
                    window['-ERROR-'].update(f'Ошибка построения графика I-V: {e}')
                window['-ERROR-'].update('')
            except Exception as e:
                window['-ERROR-'].update(str(e))
    window.close()

if __name__ == '__main__':
    main_gui()
