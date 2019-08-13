import util


def encode_image(image, text, bits):
    image_data = util.image_read(image)
    real_text = util.text_read(text)
    encode_list = util.text_to_ascii(real_text) + [0] # 0 es el último caracter.
    matrix = image_data[:, :, 0].copy()

    b_bits = util.to_binary(bits)
    b_bits = b_bits[len(b_bits)-4:]         # Los bits se pueden codificar en 4 caracteres.
    b_first = util.to_binary(matrix[0,0])   #
    matrix[0,0] = util.to_int(util.join_strings(b_first, b_bits))

    i = 0
    j = 1
    acc = ""
    for letter in encode_list: # las letras son numeros
        b_letter = util.to_binary(letter) + acc
        acc = ""
        while len(b_letter) > 0:
            encode = b_letter[len(b_letter)-bits:]
            b_letter = b_letter[:len(b_letter)-bits]
            b_value = util.to_binary(matrix[i,j])
            matrix[i,j] = util.to_int(util.join_strings(b_value,encode))
            j = j + 1

            if len(b_letter) < bits:
                acc = b_letter
                b_letter = ""

            if j >= matrix.shape[1]:
                j = 0
                i = i + 1



    # Armar la imagen usando esta matriz nueva.
    new_image = image_data.copy()
    new_image[:, :, 0] = matrix.copy()
    # Fijo la extensión
    ex_pos = len(image) - image[::-1].find('.') - 1
    new_filename = image[:ex_pos] + "out" + image[ex_pos:]
    util.image_write(new_filename, new_image)
    print("Codificación terminada, la imagen está en: " + new_filename)
