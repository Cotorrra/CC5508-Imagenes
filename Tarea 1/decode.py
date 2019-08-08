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
        if j > matrix.shape[1]:
            ++i
            j = 0
        value = matrix[i, j]
        curr_bits = acc + util.last_value(value, bits)
        if len(curr_bits) > 7:
            if curr_bits == "0000011":
                finished = True
                break
            char_list += [curr_bits[:7]]
            acc = curr_bits[7:]
        ++j

    # Convertir la lista a un string
    decoded_text = util.ascii_to_text(char_list)
    return decoded_text

