import util

def decode_image(image):
    image_data = util.image_read(image)
    # In color the first matrix is the red color
    matrix = image_data[:,:,0]
    bits = util.last_value(matrix[0,0], 4)

    finished = False;
    i = 0
    j = 1

    while(not finished):
        if j > matrix.shape[1]:
            ++i
            j=0
        value = matrix[i,j]
        #...
        if (value == 127):
            finished = True
