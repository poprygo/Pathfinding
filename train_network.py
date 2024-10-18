from generate_trainset import *
from ppix2pix import *

import os.path

from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K

from evaluate import *

from numba import cuda

image_shape_in = (512, 512, 2)
image_shape_out = (512, 512, 1)

def train_with_generator(d_model, g_model, gan_model, data_generator, n_epochs=100, n_batch=1, n_patch=16,
                         init_time=0.0, init_total_time=0.0, gpu_lock=None):
    d_losses1 = []
    d_losses2 = []
    g_losses = []
    times = []
    total_times = []

    current_time = init_time
    prev_reminder = 0

    delta_time = 10.0

    total_time = init_total_time

    # unpack dataset
    trainA, trainB = data_generator.levels_modified_X, data_generator.levels_modified_Y
    # calculate the number of batches per training epoch
    # bat_per_epo = int(len(data_denerator))
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs

    for i in tqdm(range(n_epochs)):

        for j in range(len(data_generator)):

            total_time_start = time.time()

            # select a batch of real samples
            [X_realA, X_realB], y_real = generate_real_samples(
                (data_generator.levels_modified_X, data_generator.levels_modified_Y), n_batch, n_patch)
            # generate a batch of fake samples

            if gpu_lock is not None:
                gpu_lock.acquire(True)
            X_fakeB, y_fake, d_time = generate_fake_samples(g_model, X_realA, n_patch)
            current_time += d_time
            if gpu_lock is not None:
                gpu_lock.release(True)

            if gpu_lock is not None:
                gpu_lock.acquire(True)
            start = time.time()
            # update discriminator for real samples
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # g_loss = g_model.train_on_batch(X_realA, X_realB)
            d_time = time.time() - start
            current_time += d_time
            if gpu_lock is not None:
                gpu_lock.release(True)

            total_time += time.time() - total_time_start

            # summarize performance
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))

            global callback
            # if callback is not None:
            # write_log(callback, ['g_loss'], [g_loss], i)
            # write_log(callback, ['d_loss'], [d_loss1], i)

            if ((current_time % delta_time) < prev_reminder):
                times.append(current_time)
                total_times.append(total_time)
                d_losses1.append(d_loss1)
                d_losses2.append(d_loss2)
                g_losses.append(g_loss)

            prev_reminder = current_time % delta_time

        data_generator.on_epoch_end()

    return times, d_losses1, d_losses2, g_losses, total_times


        

def train_router(cont=False):
    epochs = 2500
    # epochs = 40

    image_shape_in = (512, 512, 2)
    image_shape_out = (512, 512, 1)
    # discriminator output size. 64 for 1024x1024, 32 for 512x512, 16 for 256x256
    n_batch = 64

    # reduce batch_size to reduce memory required
    batch_size = 1

    d_model = define_discriminator(image_shape_in, image_shape_out)

    plot_model(d_model, to_file='d_model.png',
               show_shapes=True, show_layer_names=True)

    g_model = define_generator(image_shape_in, output_channels=1)

    plot_model(g_model, to_file='g_model.png',
               show_shapes=True, show_layer_names=True)

    # gan_model = define_gan(g_model, d_model, (1024, 1024, 2), intermediate_shape=(1024,1024,1))
    gan_model = define_gan(g_model, d_model, image_shape_in)

    import keras.losses

    keras.losses.custom_loss = custom_loss

    if cont is True:

        tf.keras.losses.custom_loss = custom_loss

        #if os.path.isfile('model_pix2pix_router.h5'):
        #    gan_model = load_model("model_pix2pix_router.h5")
        if os.path.isfile('model_pix2pix_router_d.h5'):
            d_model = load_model("model_pix2pix_router_d.h5")
        if os.path.isfile('model_pix2pix_router_g.h5'):
            g_model = load_model("model_pix2pix_router_g.h5", custom_objects={'custom_loss': custom_loss})
        gan_model = define_gan(g_model, d_model, image_shape_in)

    plot_model(gan_model, to_file='gan_model.png',
               show_shapes=True, show_layer_names=True)

    rooms = []
    # "working " tile set
    # with open('rooms_object_300.pickle', 'rb') as f:

    with open('rooms_object_986.pickle', 'rb') as f:
        #with open('rooms_object_632.pickle', 'rb') as f:
    #with open('rooms_object_986.pickle', 'rb') as f:
        # with open('rooms_object_986.pickle', 'rb') as f:
        # with open('rooms_object_93.pickle', 'rb') as f:
        rooms = pickle.load(f)

    gen = DesignGenerator(rooms, batch_size=batch_size, repeat=False)

    #times, d_losses1, d_losses2, g_losses, full_times, new_designs = train(
    #    d_model, g_model, gan_model, gen, epochs, batch_size, n_batch)

    times, d_losses1, d_losses2, g_losses, full_times, new_designs = train_with_reminders(
        d_model, g_model, gan_model, gen, epochs, batch_size, n_batch)


    # ax1, ax2 = plot_graphs(times, d_losses1, d_losses2,
    #                       g_losses, "1024_non_prog_gpu_time", None, None, 'x')
    
    #i removed plots
    #ax1, ax2 = plot_graphs(full_times, d_losses1, d_losses2,
    #                       g_losses, "10 samples moving avg", None, None, 'x', epochs=epochs,
    #                       times_of_new_trainsets=new_designs)

    gan_model.save('model_pix2pix_router.h5')
    d_model.save('model_pix2pix_router_d.h5')
    g_model.save('model_pix2pix_router_g.h5')

    np.savetxt('router_datapoints_noprogressive.csv',
               (times, d_losses1, d_losses2, g_losses, full_times))

    X, y = gen.__getitem__(0)

    y_pred = g_model.predict(X)

    show_map_slow(X[0], y_pred[0])
    show_map_slow(X[0], y[0])

    pass


def eval_on_data_few_models():
    load = np.load('benchmarks.npz')
    #X, Y = load['X'], load['Y']
    X = load['X']
    offset = 0

    models = [("model_pix2pix_router_g_fitted4.h5",
               "model_pix2pix_router_d_fitted4.h5"),
              ("model_pix2pix_router_g_fitted3.h5",
               "model_pix2pix_router_d_fitted3.h5"),
              ("model_pix2pix_router_g_fitted2.h5",
               "model_pix2pix_router_d_fitted2.h5"),
              ("model_pix2pix_router_g_fitted.h5",
               "model_pix2pix_router_d_fitted.h5")]

    output_merged = np.zeros((512, 512, 1))

    for g, d in models:
        d_model = load_model(d)
        g_model = load_model(g, custom_objects={
                             'custom_loss': custom_loss})
        y_pred = g_model.predict(X[offset:offset + 3])
        output_merged = np.logical_or(output_merged, y_pred[0])

    show_map_slow(X[offset], y_pred[0], file="output_merged.png")




if __name__ == "__main__":
    #train_router(True)
    #train_from_data()
    #eval_on_data_few_models()
    
    #eval_on_data()
    measure_batching()

    #render_designs()

    #train_on_hires_data(resolution = 4096)
    
    #to avoid OOM exception, we will do it by multiple runs
    
    #train_asymmetric_OOM_safe()     
    #eval_asymmetric()
    #train_different_asymmetric_models()

        #cuda.select_device(0)
        #cuda.close()


    #eval_on_hires_data()

    #train_only_on_fine_tune_set()
    #retrain_on_limited_fine_tune_set()

def measure_batching():
    if os.path.isfile('model_pix2pix_router_d_fitted.h5'):
        d_model = load_model("model_pix2pix_router_d_fitted.h5")
    if os.path.isfile('model_pix2pix_router_g_fitted.h5'):
        g_model = load_model("model_pix2pix_router_g_fitted.h5", custom_objects={'custom_loss': custom_loss})
    gan_model = define_gan(g_model, d_model, image_shape_in)
    #load = np.load('trainset_X_Y_1603940930_generation_3.npz')
    
    load = np.load('benchmarks.npz')
    #X, Y = load['X'], load['Y']
    X = load['X']
    
    X = np.concatenate((X,X,X,X,X,X,X,X,X,X))
    X = np.concatenate((X,X,X,X,X))

    #offset = 1


    #what did I mean? wtf??
    #y_pred = g_model.predict(X[offset:offset + 5])
    
    # y_pred = g_model.predict(X[5:6])
    # y_pred = g_model.predict(X[7:8])

    t = time.time()
    g_model.predict(X[0:1])
    print(time.time() - t)
    
    t = time.time()
    g_model.predict(X[0:2])
    print(time.time() - t)

    t = time.time()
    g_model.predict(X[0:4])
    print(time.time() - t)

    t = time.time()
    g_model.predict(X[0:8])
    print(time.time() - t)

    t = time.time()
    g_model.predict(X[0:16])
    print(time.time() - t)

    t = time.time()
    g_model.predict(X[0:32])
    print(time.time() - t)

    X = X[:64]
    t = time.time()
    g_model.predict(X)
    print(time.time() - t)

    
def render_designs():
    with open('rooms_object_986.pickle', 'rb') as f:
        rooms = pickle.load(f)
        for i in range(10):
            X, Y = merge_level(make_level(
                rooms, (16, 16)))

            show_map_slow(X, Y)

def eval_asymmetric():

    image_shape_in = (4096, 4096, 2)
    image_shape_out = (512, 512, 1)

    if os.path.isfile('model_pix2pix_router_d_asymmetric.h5'):
        d_model = load_model("model_pix2pix_router_d_asymmetric.h5")
    else:
        d_model = define_asymmetric_discriminator(
            image_shape_in, image_shape_out)

    if os.path.isfile('model_pix2pix_router_g_asymmetric.h5'):
        g_model = load_model("model_pix2pix_router_g_asymmetric.h5", custom_objects={
                             'custom_loss': custom_loss})
    else:
        g_model = define_asymmetric_generator(
            image_shape_in, image_shape_out, output_channels=1)

    gan_model = define_gan(g_model, d_model, image_shape_in)

    load = np.load('fitted_trainset_asymmetric_4096_512.npz')
    X, Y = load['X'], load['Y']

    #generating the random input data to check if patterns are stable

    possible_tiles = np.array([(0, 0), (0, 1), (1, 0)])

    index = np.random.choice(possible_tiles.shape[0], size=(
        4096, 4096), p=[0.98, 0.015, 0.005])

    X[0] = possible_tiles[index]

    
    #X[0] = np.random.permutation(X[0])
    y_pred = g_model.predict(X)

    #normalizing the output...
    m = np.max(y_pred[0])
    y_pred[0] = y_pred[0] / m

    show_map_slow(np.zeros((512,512,2)), y_pred[0])


    a = 1

    pass

def train_different_asymmetric_models():
    resolutions = [(4096,512), (2048,512), (1024,512), (512,512)]

    for ip, op in resolutions:
        train_asymmetric_OOM_safe(ip, op)

    pass

def train_asymmetric_OOM_safe(input_resolution, output_resolution, epochs = 2000):
    safe_period = 15

    load = np.load('fitted_trainset_asymmetric_{}_{}.npz'.format(input_resolution, output_resolution))
    X, Y = load['X'], load['Y']

    #possible_tiles = np.array([(0, 0), (0, 1), (1, 0)])
    #possible_ytiles = np.array([(0,), (1,)])

    #index = np.random.choice(possible_tiles.shape[0], size=(
    #    4096, 4096), p=[0.98, 0.015, 0.005])

    #for i in range(4):
    #    index = np.random.choice(possible_tiles.shape[0], size=(
    #        4096, 4096), p=[0.98, 0.015, 0.005])
    #    X[i] = possible_tiles[index]

    #    index = np.random.choice(possible_ytiles.shape[0], size=(
    #        512, 512), p=[0.98, 0.02])
    #    Y[i] = possible_ytiles[index]

    for i in range(epochs // safe_period):
        K.clear_session()
        try:

            
            train_asymmetric(resolution_in=input_resolution,
                             resolution_out=output_resolution, n_periods=safe_period)
        except:
            print("Exception :(")

def train_asymmetric(resolution_in = 2048, resolution_out = 512, n_periods = 100, trainset_X = None, trainset_Y = None):
    image_shape_in = (resolution_in, resolution_in, 2)
    image_shape_out = (resolution_out, resolution_out, 1)
    
    batch_size = 1

    
    if os.path.isfile('model_pix2pix_router_d_asymmetric_{}_{}.h5'.format(resolution_in, resolution_out)):
        d_model = load_model("model_pix2pix_router_d_asymmetric_{}_{}.h5".format(
            resolution_in, resolution_out))
    else:
        d_model = define_asymmetric_discriminator(
            image_shape_in, image_shape_out, filters_front = 20)

    if os.path.isfile('model_pix2pix_router_g_asymmetric_{}_{}.h5'.format(resolution_in, resolution_out)):
        g_model = load_model("model_pix2pix_router_g_asymmetric_{}_{}.h5".format(
            resolution_in, resolution_out), custom_objects={
                             'custom_loss': custom_loss})
    else:
        g_model = define_asymmetric_generator(
            image_shape_in, image_shape_out, output_channels=1, filters_front=20)

    gan_model = define_gan(g_model, d_model, image_shape_in)

    

    if (trainset_X is None) or (trainset_Y is None):

        load = np.load('fitted_trainset_asymmetric_{}_{}.npz'.format(resolution_in, resolution_out))
        X, Y = load['X'], load['Y']

    else:
        X, Y = trainset_X, trainset_Y

    

    train_fixed_data(d_model, g_model, gan_model, X, Y, n_periods, n_patch=32)

    d_model.save('model_pix2pix_router_d_asymmetric_{}_{}.h5'.format(resolution_in, resolution_out))
    g_model.save('model_pix2pix_router_g_asymmetric_{}_{}.h5'.format(resolution_in, resolution_out))

    pass

def eval_on_hires_data(resolution = 4096):
    if os.path.isfile('model_pix2pix_router_g_hires.h5'):
        g_model = load_model("model_pix2pix_router_g_hires.h5", custom_objects={
                             'custom_loss': custom_loss})

    load = np.load('fitted_trainset_large{}.npz'.format(resolution))
    X, Y = load['X'], load['Y']

    y_pred = g_model.predict(X[:1])
    show_map_slow(X[0], y_pred[0])
    pass

def train_only_on_fine_tune_set():
    image_shape_in = (1024, 1024, 2)
    image_shape_out = (1024, 1024, 1)

    batch_size = 1
    
    d_model = define_discriminator(
        image_shape_in, image_shape_out)

    plot_model(d_model, to_file='d_model.png',
               show_shapes=True, show_layer_names=True)

    g_model = define_generator(
        image_shape_in, output_channels=1)

    plot_model(g_model, to_file='g_model.png',
               show_shapes=True, show_layer_names=True)

    gan_model = define_gan(g_model, d_model, image_shape_in)

    #load = np.load('trainset_X_Y_1603940930_generation_3.npz')
    load = np.load('fine_tune_rt1.npz')
    X, Y = load['X'], load['Y']

    train_fixed_data(d_model, g_model, gan_model, X, Y, 5000, n_patch=32)

    d_model.save('model_pix2pix_router_d_fitted_only.h5')
    g_model.save('model_pix2pix_router_g_fitted_only.h5')

    #loading test data
    load = np.load('benchmarks.npz')
    X = load['X']

    y_pred = g_model.predict(X[:3])
    show_map_slow(X[1], y_pred[1], file="train_only_on_fine_tune_set.png")
    pass

def retrain_on_limited_fine_tune_set():
    
    if os.path.isfile('model_pix2pix_router_d_fitted4.h5'):
        d_model = load_model("model_pix2pix_router_d_fitted4.h5")
    if os.path.isfile('model_pix2pix_router_g_fitted4.h5'):
        g_model = load_model("model_pix2pix_router_g_fitted4.h5", custom_objects={
                             'custom_loss': custom_loss})
    gan_model = define_gan(g_model, d_model, image_shape_in)


    #load = np.load('trainset_X_Y_1603940930_generation_3.npz')
    load = np.load('bench_1.npz')
    X, Y = load['X'], load['Y']

    train_fixed_data(d_model, g_model, gan_model, X, Y, 1000, n_patch=32)

    d_model.save('model_pix2pix_router_d_fitted_on_one.h5')
    g_model.save('model_pix2pix_router_g_fitted_on_one.h5')

    #loading test data
    load = np.load('benchmarks.npz')
    X = load['X']

    y_pred = g_model.predict(X[:3])
    show_map_slow(X[1], y_pred[1], file="retrain_on_limited_fine_tune_set.png")
    pass

    pass


def train_on_hires_data(resolution=4096, filters_base = 64):

    import tensorflow as tf

    tf.config.experimental.set_lms_enabled(True)

    image_shape_in = (resolution, resolution, 2)
    image_shape_out = (resolution, resolution, 1)
    # discriminator output size. 64 for 1024x1024, 32 for 512x512, 16 for 256x256
    
    n_batch = resolution // 16


    # reduce batch_size to reduce memory required
    batch_size = 1


    dtype = 'float16'
    #K.set_floatx(dtype)

    # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
    #K.set_epsilon(1e-4)

    

    if os.path.isfile('model_pix2pix_router_d_hires{}.h5'.format(resolution)):
        d_model = load_model("model_pix2pix_router_d_hires{}.h5".format(resolution))
    else:
        d_model = define_discriminator(
            image_shape_in, image_shape_out, filters_base=filters_base)

    plot_model(d_model, to_file='d_model.png',
               show_shapes=True, show_layer_names=True)

    #d_model.summary()

    if os.path.isfile('model_pix2pix_router_g_hires{}.h5'.format(resolution)):
        g_model = load_model("model_pix2pix_router_g_hires{}.h5".format(resolution), custom_objects={
                             'custom_loss': custom_loss})
    else:
        g_model = define_generator(
            image_shape_in, output_channels=1, filters_base=filters_base)

    plot_model(g_model, to_file='g_model.png',
               show_shapes=True, show_layer_names=True)
    
    g_model.summary()

    # gan_model = define_gan(g_model, d_model, (1024, 1024, 2), intermediate_shape=(1024,1024,1))
    gan_model = define_gan(g_model, d_model, image_shape_in)

    import keras.losses

    keras.losses.custom_loss = custom_loss
    load = np.load('fitted_trainset_large{}.npz'.format(resolution))
    X, Y = load['X'], load['Y']

    #X = np.array([X[0]])
    #Y = np.array([Y[0]])

    train_fixed_data(d_model, g_model, gan_model, X, Y, 2000, n_patch=n_batch)

    d_model.save('model_pix2pix_router_d_hires.h5')
    g_model.save('model_pix2pix_router_g_hires.h5')


    y_pred = g_model.predict(X[:2])
    show_map_slow(X[0], y_pred[0])
    show_map_slow(X[0], Y[0])

def train_from_data():
    if os.path.isfile('model_pix2pix_router_d_fitted4.h5'):
        d_model = load_model("model_pix2pix_router_d_fitted4.h5")
    if os.path.isfile('model_pix2pix_router_g_fitted4.h5'):
        g_model = load_model("model_pix2pix_router_g_fitted4.h5", custom_objects={'custom_loss': custom_loss})
    gan_model = define_gan(g_model, d_model, image_shape_in)

    #load = np.load('trainset_X_Y_1603940930_generation_3.npz')
    load = np.load('fitted_trainset_2.npz')
    X, Y = load['X'], load['Y']

    train_fixed_data(d_model, g_model, gan_model, X, Y, 5000, n_patch=32)

    d_model.save('model_pix2pix_router_d_fitted4.h5')
    g_model.save('model_pix2pix_router_g_fitted4.h5')


    y_pred = g_model.predict(X[:3])
    show_map_slow(X[0], y_pred[0])
# eval()

def eval_on_data():
    if os.path.isfile('model_pix2pix_router_d_fitted.h5'):
        d_model = load_model("model_pix2pix_router_d_fitted.h5")
    if os.path.isfile('model_pix2pix_router_g_fitted.h5'):
        g_model = load_model("model_pix2pix_router_g_fitted.h5", custom_objects={'custom_loss': custom_loss})
    gan_model = define_gan(g_model, d_model, image_shape_in)
    #load = np.load('trainset_X_Y_1603940930_generation_3.npz')
    
    load = np.load('benchmarks.npz')
    #X, Y = load['X'], load['Y']
    X = load['X']
    

    #offset = 1


    #what did I mean? wtf??
    #y_pred = g_model.predict(X[offset:offset + 5])
    
    y_pred = g_model.predict(X)

    np.savez_compressed("for_clustering_debug2.npz", X = X[1], Y = y_pred[1])
    show_map_slow(X[1], y_pred[1])


    # import benchmark_format

    # X = benchmark_format.parse_benchmark("./benches/RT01.inp")
    # y_pred = g_model.predict(np.array([X,X]))
    # show_map_slow(X, y_pred[0])


def load_and_evaluate():
    with open('eval.npy', 'rb') as f:
        X = np.load(f)
        y_pred = np.load(f)
        y = np.load(f)

        # for debug
        #show_map_slow(X, y_pred)

        evaluate_output(X, y_pred)

# load_and_evaluate()
