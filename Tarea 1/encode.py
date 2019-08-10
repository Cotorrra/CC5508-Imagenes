import util


def encode_image(image, text, bits):
    image_data = util.image_read(image)
    encode_list = util.text_to_ascii(text)
    matrix = image_data[:, :, 0]

    b_bits = util.to_binary(bits)
    b_first = util.to_binary(matrix[0,0])
    matrix[0,0] = util.to_int(util.join_strings(b_first, b_bits))

    i = 0
    j = 1
    acc = ""
    for letter in encode_list: # las letras son numeros
        b_letter = util.to_binary(letter) + acc
        acc = ""
        while b_letter != "":
            if j > matrix.shape[1]:
                j=0
                ++i

            encode = b_letter[:bits]
            b_letter = b_letter[bits:]
            b_value = util.to_binary(matrix[i,j])
            matrix[i,j] = util.to_int(util.join_strings(b_value,encode))

            ++j
            if len(b_letter) <= bits:
                acc = b_letter
                b_letter = ""

    # Armar la imagen usando esta matriz nueva.
    new_image = image_data
    new_image[:, :, 0] = matrix
    # Fijo la extensión
    ex_pos = len(image) - image[::-1].find('.') - 1
    new_filename = image[:ex_pos] + "out" + image[ex_pos:]
    util.image_write(new_filename, new_image)
