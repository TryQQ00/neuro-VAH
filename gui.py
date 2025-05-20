import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
from parser import VerilogAParser
from simulator import DeviceModel
from generator import SignalGenerator
from rnn_model import LSTMModel, train_model
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_figure(canvas, figure):
    # Очищаем Canvas
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, master=canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def main_gui():
    sg.theme('LightBlue3')
    layout = [
        [sg.Text('Verilog-A файл:'), sg.Input(key='-VAFILE-', size=(60,1)), sg.FileBrowse(file_types=(('VA','*.va'),))],
        [sg.Text('Vmin:'), sg.Input('0.0', key='-VMIN-', size=(8,1)),
         sg.Text('Vmax:'), sg.Input('1.0', key='-VMAX-', size=(8,1)),
         sg.Text('Samples:'), sg.Input('100', key='-SAMPLES-', size=(8,1)),
         sg.Text('Epochs:'), sg.Input('50', key='-EPOCHS-', size=(8,1))],
        [sg.Button('Запустить обучение', key='-RUN-')],
        [sg.Text('Loss (MSE):')],
        [sg.Canvas(key='-LOSS-CANVAS-', size=(600,300), background_color='white', expand_x=True, expand_y=True)],
        [sg.Text('I-V Curve:')],
        [sg.Canvas(key='-IV-CANVAS-', size=(600,300), background_color='white', expand_x=True, expand_y=True)],
        [sg.Text('', key='-ERROR-', size=(100,2), text_color='red')]
    ]
    window = sg.Window(
        'Neuro-VAH', layout, finalize=True, resizable=True, element_justification='center', auto_size_text=True, auto_size_buttons=True,
        size=(1100, 800)
    )
    window.TKroot.minsize(900, 600)
    loss_fig_agg = None
    iv_fig_agg = None
    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, None):
            break
        if event == '-RUN-':
            try:
                va_file = values['-VAFILE-']
                vmin = float(values['-VMIN-'])
                vmax = float(values['-VMAX-'])
                samples = int(values['-SAMPLES-'])
                epochs = int(values['-EPOCHS-'])
                parser = VerilogAParser()
                info = parser.parse(va_file)
                params = info['params']
                t, v = SignalGenerator.sweep(vmin, vmax, samples)
                device = DeviceModel(params, 'Диод')
                i_phys = device.simulate(v)
                # Проверка на NaN/inf
                if not np.all(np.isfinite(i_phys)):
                    raise ValueError('В симуляции физической модели получены некорректные значения (NaN/inf). Проверьте параметры и диапазон напряжений.')
                X = torch.tensor(v[None, :, None], dtype=torch.float32)
                Y = torch.tensor(i_phys[None, :, None], dtype=torch.float32)
                model = LSTMModel()
                device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model, losses = train_model(model, device_torch, X.to(device_torch), Y.to(device_torch), epochs=epochs)
                # График loss
                try:
                    fig1 = plt.figure(figsize=(7,3))
                    plt.plot(losses)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training Loss')
                    plt.tight_layout()
                    if loss_fig_agg:
                        for child in window['-LOSS-CANVAS-'].TKCanvas.winfo_children():
                            child.destroy()
                    loss_fig_agg = draw_figure(window['-LOSS-CANVAS-'].TKCanvas, fig1)
                    plt.close(fig1)
                except Exception as e:
                    window['-ERROR-'].update(f'Ошибка построения графика Loss: {e}')
                # Сравнение ВАХ
                try:
                    model.eval()
                    with torch.no_grad():
                        y_pred = model(X.to(device_torch)).cpu().numpy().reshape(-1)
                    if not np.all(np.isfinite(y_pred)):
                        raise ValueError('В предсказании нейросети получены некорректные значения (NaN/inf).')
                    fig2 = plt.figure(figsize=(7,3))
                    plt.plot(v, i_phys, label='Reference')
                    plt.plot(v, y_pred, '--', label='Predicted')
                    plt.xlabel('V')
                    plt.ylabel('I')
                    plt.legend()
                    plt.title('I-V Curve')
                    plt.tight_layout()
                    if iv_fig_agg:
                        for child in window['-IV-CANVAS-'].TKCanvas.winfo_children():
                            child.destroy()
                    iv_fig_agg = draw_figure(window['-IV-CANVAS-'].TKCanvas, fig2)
                    plt.close(fig2)
                except Exception as e:
                    window['-ERROR-'].update(f'Ошибка построения графика I-V: {e}')
                window['-ERROR-'].update('')
            except Exception as e:
                window['-ERROR-'].update(str(e))
        # Перерисовка при изменении размера окна
        if event == '__TIMEOUT__':
            if loss_fig_agg:
                try:
                    loss_fig_agg.get_tk_widget().config(width=window['-LOSS-CANVAS-'].TKCanvas.winfo_width(), height=window['-LOSS-CANVAS-'].TKCanvas.winfo_height())
                except Exception:
                    pass
            if iv_fig_agg:
                try:
                    iv_fig_agg.get_tk_widget().config(width=window['-IV-CANVAS-'].TKCanvas.winfo_width(), height=window['-IV-CANVAS-'].TKCanvas.winfo_height())
                except Exception:
                    pass
    window.close()