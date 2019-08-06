import argparse

def encode(img, txt):
    print('Woy a encodear: '+img+' con lo siguiente: '+txt)

def decode(file):
    print('Woy a decodear: '+file)


'''

--encode: indica que la tarea es codificar un texto de entrada dentrode una imagen. 
La imagen resultante tiene el mismo nombre que el de la imagen de entrada más el prefijo out. 
La extensión es la misma que la de la imagen de entrada.
[Por ahora sólo debo llamar a encode, con los argumentos necesarios.]

--decode: indica que la tarea es decodificar una imagen, en este caso se mostrará el texto oculto.
[Por ahora solo debo llamar a decode con los argumentos necesarios.]

--nbits <N>: indica el número de bits menos significativos a ser usado.
N toma un valor entre entre 1 y 8, inclusive. Este argumento esusado en el modo encode.

--image <image filename>: indica la imagen sobre la que se codificará un texto.  
Pueden suponer que la imagen está en escala de grises.

--text<text  filename>: indica el archivo que contiene el texto a codificar. 
Este argumento es usado en el modoenconde.
'''


parser = argparse.ArgumentParser(description='Procesa imágenes para codificar/decodificar un mensaje dentro de ésta.')

parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()

print(args.accumulate(args.integers))