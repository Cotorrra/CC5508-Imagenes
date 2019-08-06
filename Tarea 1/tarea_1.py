import argparse

def encode(img, txt):
    print('Woy a encodear: '+img+' con lo siguiente: '+txt)

def decode(file):
    print('Woy a decodear: '+file)

parser = argparse.ArgumentParser(description='Procesa imágenes para codificar/decodificar un mensaje dentro de ésta.')

parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()

print(args.accumulate(args.integers))