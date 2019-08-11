
import argparse
import encode
import decode

'''
--encode: indica que la tarea es codificar un texto de entrada dentrode una imagen. 
La imagen resultante tiene el mismo nombre que el de la imagen de entrada más el prefijo out. 
La extensión es la misma que la de la imagen de entrada.

--decode: indica que la tarea es decodificar una imagen, en este caso se mostrará el texto oculto.

--nbits <N>: indica el número de bits menos significativos a ser usado.
N toma un valor entre entre 1 y 8, inclusive. Este argumento esusado en el modo encode.

--image <image filename>: indica la imagen sobre la que se codificará un texto.  
Pueden suponer que la imagen está en escala de grises.

--text<text  filename>: indica el archivo que contiene el texto a codificar. 
Este argumento es usado en el modoenconde.
'''


parser = argparse.ArgumentParser(description='Lo k ase')

command_group = parser.add_mutually_exclusive_group()
command_group.add_argument('--encode', help='Codifica la imagen dada', action="store_true")
command_group.add_argument('--decode', help='Decodifica la imagen dada', action="store_true")
parser.add_argument('--image', action="store", type=str, nargs=1, required=True,
                    help='Indica la imagen sobre la que se codificará/decodificará un texto')
parser.add_argument('--text', action="store", type=str, nargs=1, default="",
                    help='Indica el archivo que contiene el texto a codificar.')
parser.add_argument('--nbits', action="store", type=int, nargs=1, default=1,
                    help='Cantidad de bits menos significativos en la que se va codificar')

args = parser.parse_args()
image = args.image[0]

if args.encode:
    text = args.text[0]
    bits = args.nbits[0]
    if bits > 8 or bits < 1:
        raise ValueError("--nbits tiene que ser un entero entre 1 y 8")

    encode.encode_image(image, text, bits)
    exit(0)

if args.decode:
    print(decode.decode_image(image))
    exit(0)