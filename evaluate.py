import numpy as np
import networkx as nx
import random
import math

from functools import reduce

from train_network import *

from generate_trainset import *
from ppix2pix import *

import os.path

from tensorflow.keras.models import load_model

from scale_design import *



def evaluate_on_tiled_design(rooms):
    tf.keras.losses.custom_loss = custom_loss

    gan_model = None
    d_model = None
    g_model = None

    if os.path.isfile('model_pix2pix_router.h5'):
        gan_model = load_model("model_pix2pix_router.h5")
    if os.path.isfile('model_pix2pix_router_d.h5'):
        d_model = load_model("model_pix2pix_router_d.h5")
    if os.path.isfile('model_pix2pix_router_g.h5'):
        g_model = load_model("model_pix2pix_router_g.h5", custom_objects={
            'custom_loss': custom_loss})

    # rooms = rooms

    # with open('rooms_object_300.pickle', 'rb') as f:
    #    rooms = CustomUnpickler(f).load()

    gen = DesignGenerator(rooms, batch_size=5, repeat=False)

    X = gen.__getitem__(0)

    y_pred = g_model.predict(X)
    show_map_slow(X[0], y_pred[0])


def evaluate_on_generated_design(design=None):
    tf.keras.losses.custom_loss = custom_loss

    image_shape_in = (512, 512, 2)
    image_shape_out = (512, 512, 1)

    d_model = load_model("model_pix2pix_router_d_fitted.h5")
    g_model = load_model("model_pix2pix_router_g_fitted.h5", custom_objects={'custom_loss': custom_loss})

    gan_model = define_gan(g_model, d_model, image_shape_in)

    if design is None:

        content_shape_x = 128
        content_shape_y = 128

        X_background = np.zeros((512, 512, 2))
        X = generate_map(resolution=(
            content_shape_x, content_shape_y), pins=128, connection_pins=0, obstacles=10, fraction=0.2, routed=False)

        X = np.array(X)
        shift_x = 0
        shift_y = 0
        X_background[shift_x:content_shape_x + shift_x, shift_y:content_shape_y + shift_y] = X

        X = np.expand_dims(X_background, 0)

    else:
        X = np.expand_dims(design, 0)

    y_pred = g_model.predict(X)
    show_map_slow(X[0], y_pred[0], threshold=0.3, file = "output.png")


def evaluate_output(design, output):
    g = nx.Graph()

    width, height = design.shape[:2]

    pins = []

    for x in range(width):
        for y in range(height):
            if design[x][y][1] > 0.5:
                pins.append((x, y))
            if output[x][y][0] > 0.5 and design[x][y][0] < 0.5:
                g.add_node((x, y))
            if x > 0:
                if output[x - 1][y][0] > 0.5 and design[x - 1][y][0] < 0.5:
                    g.add_edge((x - 1, y), (x, y))
            if y > 0:
                if output[x][y - 1][0] > 0.5 and design[x - 1][y - 1][0] < 0.5:
                    g.add_edge((x, y - 1), (x, y))

    print('Total design size = {}'.format(width * height))

    # graph is built, now evaluating it
    # taking the largest component
    components = list(nx.connected_components(g))
    components.sort(reverse=True, key=lambda x: len(x))
    component = components[0]

    routed_pins = reduce((lambda x, y: x + y), map(lambda node: 1 if (node in component) else 0, pins))

    print("Wire length: {}, pins routed: {}%".format(len(component), routed_pins / len(pins) * 100))

def scale_up(wires):
    X = np.zeros((512,512,2))
    _, scaled_wires = scale(X, wires)
    return scaled_wires


def evaluate_fine_tuned_model_on_real_benchmark():
    
    
    pass

if __name__ == "__main__":
    import benchmark_format

    X = benchmark_format.parse_benchmark("./benches/RT01.inp")
    show_map_slow(X)
    #evaluate_on_generated_design(X)
    #evaluate_on_generated_design()
