import numpy as np
import re
import math
import random
import generate_trainset
import generate_trainset

def push(pin_position, obstacle_map):
    if obstacle_map[pin_position[0], pin_position[1], 0] == 0.0:
        # pin is ok
        return pin_position
    
    #there are few categories of obstacled pins
    # oo
    # po
    # oo
    #in this case, pushing left
    if (obstacle_map[pin_position[0] + 1, pin_position[1], 0] == 1.0) and (obstacle_map[pin_position[0], pin_position[1] + 1, 0] == 1.0) and (obstacle_map[pin_position[0], pin_position[1] - 1, 0] == 1.0):
        return pin_position[0] - 1, pin_position[1]

    #there are few categories of obstacled pins
    # oo
    # op
    # oo
    #in this case, pushing right
    if obstacle_map[pin_position[0] - 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0], pin_position[1] + 1, 0] == 1.0 and obstacle_map[pin_position[0], pin_position[1] - 1, 0] == 1.0:
        return pin_position[0] + 1, pin_position[1]

    #there are few categories of obstacled pins
    # ooo
    # opo

    #in this case, pushing down
    if obstacle_map[pin_position[0] - 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0] + 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0], pin_position[1] - 1, 0] == 1.0:
        return pin_position[0], pin_position[1] + 1

    #there are few categories of obstacled pins
    
    # opo
    # ooo

    #in this case, pushing up
    if obstacle_map[pin_position[0] - 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0] + 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0], pin_position[1] + 1, 0] == 1.0:
        return pin_position[0], pin_position[1] - 1
    
    #now corner cases

    # po
    # oo
    #in this case, pushing left and up
    if obstacle_map[pin_position[0] + 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0], pin_position[1] + 1, 0] == 1.0:
        return pin_position[0] - 1, pin_position[1] - 1

    # oo
    # po

    #in this case, pushing left and down
    if obstacle_map[pin_position[0] + 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0], pin_position[1] - 1, 0] == 1.0:
        return pin_position[0] - 1, pin_position[1] + 1

    # oo
    # op

    #in this case, pushing right and down
    if obstacle_map[pin_position[0] - 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0], pin_position[1] - 1, 0] == 1.0:
        return pin_position[0] + 1, pin_position[1] + 1

    # op
    # oo

    #in this case, pushing right and up
    if obstacle_map[pin_position[0] - 1, pin_position[1], 0] == 1.0 and obstacle_map[pin_position[0], pin_position[1] + 1, 0] == 1.0:
        return pin_position[0] + 1, pin_position[1] - 1

    return pin_position

def get_X_Y_from_bench(filename_ip, filename_op, max_res = (512,512)):
    state = None


    num_pins = 0

    num_obs = 0

    pins = []
    pins_extra = []
    obstacles = []

    design_name = filename_ip
    output_name = filename_op

    with open(design_name, "r") as f:
        lines = f.readlines()
        num_pins = 0
        for l in lines:
            match = re.fullmatch(r'^\d+\n$', l)
            if match is not None:
                if state is None:
                    state = "PINS"
                elif state == "PINS":
                    state = "OBS"
                continue
                # num_pins = int(m.group(1))

            # match = re.match(r'k (\d+)', l)
            # if match is not None:
            #     state = "OBS"
            #     num_obs = int(m.group(1))

            if state == "PINS":
                match = re.match(r'(\d+) (\d+)\n', l)
                if match is not None:
                    pin = [int(match.group(1)), int(match.group(2))]
                    pins.append(pin)
                    continue

            if state == "OBS":
                match = re.match(r'(\d+) (\d+) (\d+) (\d+)\n', l)
                if match is not None:
                    obs = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                    obstacles.append(obs)
                    continue

    wires = []

    state = None

    if output_name is not None:

        with open(output_name, "r") as f:
            lines = f.readlines()
            num_pins = 0
            for l in lines:
                match = re.fullmatch(r'^\d+\n$', l)
                if match is not None:
                    if state is None:
                        state = "PINS"
                    elif state == "PINS":
                        state = "OBS"
                    continue
                    # num_pins = int(m.group(1))

                # match = re.match(r'k (\d+)', l)
                # if match is not None:
                #     state = "OBS"
                #     num_obs = int(m.group(1))

                if state == "PINS":
                    match = re.match(r'(\d+) (\d+)\n', l)
                    if match is not None:
                        pin = [int(match.group(1)), int(match.group(2))]
                        pins_extra.append(pin)
                        continue

                if state == "OBS":
                    match = re.match(r'(\d+) (\d+) (\d+) (\d+)\n', l)
                    if match is not None:
                        obs = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                        wires.append(obs)
                        continue

    # mapping this pins and obstacles

    max_x, min_x = 0, 0
    max_y, min_y = 0, 0

    for p in pins:
        if p[0] < min_x: min_x = p[0]
        if p[0] > max_x: max_x = p[0]
        if p[1] < min_y: min_y = p[1]
        if p[1] > max_y: max_y = p[1]

    for o in obstacles:
        if o[0] < min_x:
            min_x = o[0]
        if o[2] > max_x:
            max_x = o[2]
        if o[1] < min_y:
            min_y = o[1]
        if o[3] > max_y:
            max_y = o[3]

    for w in wires:
        if w[0] < min_x: min_x = w[0]
        if w[1] > max_x: max_x = w[1]
        if w[2] < min_y: min_y = w[2]
        if w[3] > max_y: max_y = w[3]

    design = np.zeros((max_res[0], max_res[1], 2))
    route = np.zeros((max_res[0], max_res[1], 1))

    max_res = (max_res[0] - 1, max_res[1] - 1)

    for p in pins:
        p[0] = int((p[0] - min_x) / (max_x - min_x) * max_res[0])
        p[1] = int((p[1] - min_y) / (max_y - min_y) * max_res[1])

    for o in obstacles:
        o[0] = int((o[0] - min_x) / (max_x - min_x) * max_res[0])
        o[1] = math.floor((o[1] - min_y) / (max_y - min_y) * max_res[1])
        o[2] = int((o[2] - min_x) / (max_x - min_x) * max_res[0])
        o[3] = math.floor((o[3] - min_y) / (max_y - min_y) * max_res[1])

    for w in wires:
        w[0] = int((w[0] - min_x) / (max_x - min_x) * max_res[0])
        w[1] = math.floor((w[1] - min_y) / (max_y - min_y) * max_res[1])
        w[2] = int((w[2] - min_x) / (max_x - min_x) * max_res[0])
        w[3] = math.floor((w[3] - min_y) / (max_y - min_y) * max_res[1])

    for o in obstacles:
        for x in range(o[0], o[2] + 1):
            for y in range(o[1], o[3] + 1):
                design[x, y, 0] = 1.0

    for pin in pins:
        if pin[0] <= max_res[0] and pin[1] <= max_res[1]:
            #shifted_pin = push(pin, design)
            #design[shifted_pin[0], shifted_pin[1], 1] = 1
            design[pin[0], pin[1], 1] = 1
            design[pin[0], pin[1], 0] = 0

    design_with_extra_pins = np.array(design)
    for pin in pins_extra:
        if pin[0] <= max_res[0] and pin[1] <= max_res[1]:
            #shifted_pin = push(pin, design)
            #design[shifted_pin[0], shifted_pin[1], 1] = 1
            design_with_extra_pins[pin[0], pin[1], 1] = 1
            design_with_extra_pins[pin[0], pin[1], 0] = 0
            design[pin[0], pin[1], 0] = 0
    

    for w in wires:
        for x in range(w[0], w[2] + 1):
            for y in range(w[1], w[3] + 1):
                #№x_s, y_s = push((x,y), design)
                #route[x_s, y_s, 0] = 1.0
                route[x, y, 0] = 1.0
                design[x, y, 0] = 0

    print(np.sum(design[:, :, 0]))
    print(np.sum(design[:, :, 1]))

    return design, route, design_with_extra_pins

def rotate(X,Y):
    levels_modified_X = [X,]
    levels_modified_Y = [Y,]

    for k in range(1, 4):
        X = np.rot90(X, k=k)
        Y = np.rot90(Y, k=k)
        levels_modified_X.append(X)
        levels_modified_Y.append(Y)

        # show_map_slow(shifted_X,shifted_Y)
        pass

    print("Number of tiles in rotate function: {}".format(np.sum(levels_modified_Y[0])))
    return levels_modified_X, levels_modified_Y

def cut_design_into_tiles(X, Y, n, sizes = ( (256,256), ), maxres = 512):

    trainset_X = []
    trainset_Y = []

    rotated_X, rotated_Y = rotate(X, Y)
    trainset_X += rotated_X
    trainset_Y += rotated_Y




    for size in sizes:
        for j in range(n):
            random_shift_x = random.randint(0, maxres - size[0] - 1)
            random_shift_y = random.randint(0, maxres - size[1] - 1)

            shifted_X = np.zeros(
                (maxres , maxres, 2))
            shifted_Y = np.zeros((maxres, maxres, 1))


            shifted_X[random_shift_x:random_shift_x + size[0],
            random_shift_y:random_shift_y + size[1]] = X[random_shift_x:random_shift_x + size[0],
            random_shift_y:random_shift_y + size[1]]

            shifted_Y[random_shift_x:random_shift_x + size[0],
            random_shift_y:random_shift_y + size[1]] = Y[random_shift_x:random_shift_x + size[0],
                                                       random_shift_y:random_shift_y + size[1]]

            rotated_X, rotated_Y = rotate(shifted_X, shifted_Y)
            trainset_X += rotated_X
            trainset_Y += rotated_Y

    #debug: printing # of tiles
    print("Number of tiles: {}".format(np.sum(trainset_Y[0]) ))
    #generate_trainset.show_map_slow(trainset_X[0], trainset_Y[0])

    return trainset_X, trainset_Y


    # should we cut all the design to tiles? probably w e should not have any tiles at all, just cut different
    #don't forget to add pin if cropping cuts the wire

def generate_tiles_from_benches():
    benchmarks_parsed = []
    #bench_names = ["benches/RT01.inp", "benches/RT02.inp", "benches/RT03.inp", "benches/RT04.inp", "benches/RT05.inp"]
    bench_names = ["benches/RT01.inp"]
    #bench_ops = ["benches/old/RT01.tmp3", "benches/old/RT02.tmp3", "benches/old/RT03.tmp3", "benches/old/RT04.tmp3", "benches/old/RT05.tmp3"]
    bench_ops = ["benches/old/RT01.tmp3"]
    full_trainset_X = []
    full_trainset_Y = []
    for name, name2 in zip(bench_names, bench_ops):
        X, Y, X2 = get_X_Y_from_bench(name, name2)
        #generate_trainset.show_map_slow(X, Y)
        trainset_X, trainset_Y = cut_design_into_tiles(X,Y, 10)
        full_trainset_X += trainset_X
        full_trainset_Y += trainset_Y
        #trainset_X, trainset_Y = cut_design_into_tiles(X2, Y, 0)
        #full_trainset_X += trainset_X
        #full_trainset_Y += trainset_Y

        benchmarks_parsed.append(X)

    a = 1

    full_trainset_Y = np.array(full_trainset_Y)
    full_trainset_X = np.array(full_trainset_X)

    np.savez_compressed("bench_1.npz", X=full_trainset_X, Y=full_trainset_Y )
    #np.savez_compressed("benchmarks.npz", X=benchmarks_parsed)
    
    return full_trainset_X, full_trainset_Y

def generate_tiles_from_large_benchmark(resolution = 4096):
    bench_name = "benches/RT03.inp"
    bench_op = "benches/old/RT03.tmp3"
    X, Y, X2 = get_X_Y_from_bench(bench_name, bench_op, max_res=(resolution, resolution))
    trainset_X, trainset_Y = cut_design_into_tiles(X, Y, 0)
    trainset_X = np.array(trainset_X, dtype=np.float16)
    trainset_Y = np.array(trainset_Y, dtype=np.float16)

    np.savez_compressed("fitted_trainset_large{}.npz".format(resolution),
                        X=trainset_X, Y=trainset_Y)

    pass

def generate_asymmetric_trainset(input_resolution = 4096, output_resolution=512):
    bench_name = "benches/RT03.inp"
    bench_op = "benches/old/RT03.tmp3"
    
    hires_X, hires_Y, hires_X2 = get_X_Y_from_bench(bench_name, bench_op, max_res=(input_resolution, input_resolution))
    lowres_X, lowres_Y, lowres_X2 = get_X_Y_from_bench(bench_name, bench_op, max_res=(output_resolution, output_resolution))
    
    trainset_X, _ =  cut_design_into_tiles(hires_X, hires_Y, 0)
    _, trainset_Y =  cut_design_into_tiles(lowres_X, lowres_Y, 0)

    trainset_X = np.array(trainset_X)
    trainset_Y = np.array(trainset_Y)

    np.savez_compressed("fitted_trainset_asymmetric_{}_{}.npz".format(input_resolution, output_resolution), X=trainset_X, Y=trainset_Y)

    pass

def show_hires_and_lowres_reference():
    bench_name = "benches/RT05.inp"
    bench_op = "benches/old/RT05.tmp3"

    hires_X, hires_Y, hires_X2 = get_X_Y_from_bench(
        bench_name, bench_op, max_res=(1024, 1024))
    lowres_X, lowres_Y, lowres_X2 = get_X_Y_from_bench(
        bench_name, bench_op, max_res=(512, 512))

    generate_trainset.show_map_slow(hires_X, hires_Y)
    generate_trainset.show_map_slow(lowres_X, lowres_Y)




if __name__ == '__main__':
    #generate_tiles_from_benches()
    
    #generate_tiles_from_large_benchmark(2048)
    
    #resolutions = [(4096,512), (2048, 512), (1024, 512), (512, 512)]
    #for ip, op in resolutions:
    #    generate_asymmetric_trainset(ip, op)

    show_hires_and_lowres_reference()

    #X, Y = get_X_Y_from_bench("benches/RT01.inp", "benches/old/RT01.tmp3")
    #generate_trainset.show_map_slow(X, Y )

    pass
