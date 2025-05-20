from parser import VerilogAParser
import tempfile
import os

def test_parser_typical_va():
    va_text = '''
    module testmod(a, b);
    parameter real Is = 1e-14;
    parameter integer N = 1;
    endmodule
    '''
    with tempfile.NamedTemporaryFile('w+', suffix='.va', delete=False) as f:
        f.write(va_text)
        f.flush()
        parser = VerilogAParser()
        info = parser.parse(f.name)
        assert info['module'] == 'testmod'
        assert info['ports'] == ['a', 'b']
        assert info['params']['Is'] == 1e-14
        assert info['params']['N'] == 1

def test_parser_real_va_files():
    # Пути к вашим файлам
    va_files = [
        '/Users/art/Documents/extract_parameters/data/code/ASMHEMT/vacode/asmhemt.va',
        '/Users/art/Documents/extract_parameters/data/code/ASMHEMT/vacode/asmhemt101_0.va',
        '/Users/art/Documents/extract_parameters/data/code/mextram/vacode/bjt505t.va',
        '/Users/art/Documents/extract_parameters/data/code/mextram/vacode/bjt505.va',
    ]
    parser = VerilogAParser()
    for va_fp in va_files:
        if not os.path.exists(va_fp):
            continue  # Пропускаем если файла нет
        info = parser.parse(va_fp)
        assert 'module' in info
        assert 'ports' in info
        assert 'params' in info
        assert isinstance(info['params'], dict)
        # Проверяем, что параметры либо числа, либо строки
        for v in info['params'].values():
            assert isinstance(v, (float, int, str)) 