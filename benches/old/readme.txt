* OS: Linux

* Compiler: gcc 4.0.3

* for 32-bit machine

* Commands
  ./rst input_filename temporary_filename
  ./ref input_filename temporary_filename output_filename

* Input Formats
  m (# of pins)
    x1 y1
    x2 y2
    ...
    xm ym
  k (# of obstacles)
    xmin1 ymin1 xmax1 ymax1
    xmin2 ymin2 xmax2 ymax2
    ...
    xminm yminm xmaxm ymaxm

* Output Formats
  s (# of Steiner points)
    x1 y1
    x2 y2
    ...
    xs ys
  t (# of paths)
    u1 (# of points along path 1)
      x1 y1
      x2 y2
      ...
      x{u1} y{u1}
    u2 (# of points along path 2)
      ...
    ...
    ut (# of points along path t)
      ...