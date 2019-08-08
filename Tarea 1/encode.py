import util

def encode_image(image, text, bits):
    image_data = util.image_read(image)
    encode_list = util.text_to_ascii(text)
    matrix = image_data[:, :, 0]

    init = matrix[0, 0]
    b_bits = util.to_binary(bits)
    b_init = util.to_binary(matrix[0, 0])
    bit_encode = b_init[:len(b_init)-4] + b_bits[len(b_bits)-4:]
    matrix[0, 0] = util.to_int(bit_encode)

    i = 0
    j = 1
    for value in encode_list:
        print("HELLO WORLD")

