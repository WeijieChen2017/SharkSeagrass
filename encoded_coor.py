x_space_mm = 600
y_space_mm = 600
z_space_mm = 400
# we can distribute different bits to x, y, and z for different resolution requirements

x_bits = 24
y_bits = 24
z_bits = 16

assert x_bits + y_bits + z_bits == 64

def physical_space_to_integer_coordinates_v2(x_mm, y_mm, z_mm):

    # Resolution values derived from bit allocations and physical space
    resolution_x = x_space_mm / (2 ** x_bits)
    resolution_y = y_space_mm / (2 ** y_bits)
    resolution_z = z_space_mm / (2 ** z_bits)

    # Convert physical coordinates to integer representations
    x_int = int(x_mm / resolution_x)
    y_int = int(y_mm / resolution_y)
    z_int = int(z_mm / resolution_z)
    
    x_bin = bin(x_int)
    y_bin = bin(y_int)
    z_bin = bin(z_int)

    encoded = x_bin[2:] + y_bin[2:] + z_bin[2:]
    encoded = int(encoded, 2)
    return encoded

def integer_coordinates_to_physical_space_v2(encoded, x_bits=24, y_bits=24, z_bits=16):

    # take the first x_bits bits for x
    x_int = encoded >> (y_bits + z_bits)
    # take the next y_bit bits for y
    y_int = (encoded >> z_bits) & (2**y_bits - 1)
    # take the last z_bit bits for z
    z_int = encoded & (2**z_bits - 1)

    # Resolution values derived from bit allocations and physical space
    resolution_x = x_space_mm / (2 ** x_bits)
    resolution_y = y_space_mm / (2 ** y_bits)
    resolution_z = z_space_mm / (2 ** z_bits)

    # Convert integer representations to physical coordinates
    x_mm = x_int * resolution_x
    y_mm = y_int * resolution_y
    z_mm = z_int * resolution_z

    return x_mm, y_mm, z_mm

# example we have the location (300, 400, 200) in mm
x_mm, y_mm, z_mm = 300, 400, 200

assert x_mm < x_space_mm
assert y_mm < y_space_mm
assert z_mm < z_space_mm

encoded_coor = physical_space_to_integer_coordinates_v2(x_mm, y_mm, z_mm)
print("encoded_coor: ", encoded_coor)

recon_coors = integer_coordinates_to_physical_space_v2(encoded_coor)
print("recon_coors: ", recon_coors)