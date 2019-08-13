import util


def decode_image(image):
    image_data = util.image_read(image)
    # In color the first matrix is the red color
    matrix = image_data[:, :, 0]
    bits = util.last_value(matrix[0, 0], 4)

    finished = False
    i = 0
    j = 1
    acc = ''
    char_list = []
    while not finished:
        value = matrix[i, j]
        curr_bits = util.to_binary(util.last_value(value, bits))
        acc = curr_bits[len(curr_bits)-bits:] + acc
        j = j + 1

        if len(acc) >= 8:
            if util.to_int(acc[:8]) == 0:
                finished = True
            else:
                char_list = char_list + [util.to_int(acc[:8])]
                acc = acc[8:]

        if j >= matrix.shape[1]:
            i = i + 1
            j = 0


    # Convertir la lista a un string
    decoded_text = util.ascii_to_text(char_list)
    return decoded_text

